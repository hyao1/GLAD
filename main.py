#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from torch.utils.data import DataLoader
from pipeline import StableDiffusionPipeline
from ddim_scheduling import DDIMScheduler
import copy
import numpy as np
import argparse
import logging
import os
import time
import torch
import torch.utils.checkpoint
from torchvision import transforms, utils
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from tqdm.auto import tqdm
from transformers import AutoTokenizer
from diffusers.optimization import get_scheduler
import bitsandbytes as bnb

from dataset.dataset import MVTecDataset
from creat_model import model
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

from utilize.utilize import normalize, fix_seeds, compute_pro, reconstruction

from creat_model import get_vit_encoder
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d

import warnings
warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--train", type=bool)

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="CompVis/stable-diffusion-v1-4",
                        help="Path to pretrained model or model identifier.", )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )

    # dataset setting
    parser.add_argument("--instance_data_dir", type=str, default="/hdd/Datasets/MVTec-AD")
    parser.add_argument("--anomaly_data_dir", type=str, default="/hdd/Datasets/dtd")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--class_name", type=str)
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--denoise_step", type=int, default=500)
    parser.add_argument("--min_step", type=int, default=350)
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks")
    parser.add_argument("--resolution", type=int, default=512, )
    parser.add_argument("--dino_resolution", type=int, default=512, )
    parser.add_argument("--v", type=int, default=0, )
    parser.add_argument("--input_threshold", type=float, default=0.0, )
    parser.add_argument("--dino_save_path", default=None, type=str)

    parser.add_argument("--inference_step", type=int, default=25, )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=2, help="Batch size (per device) for sampling images.")

    # train setting
    parser.add_argument("--checkpointing_steps", type=int, default=200, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform.", )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant", help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"]'), )
    parser.add_argument("--lr_warmup_steps", type=int, default=500,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8, help=(
        "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."), )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--lr_num_cycles", type=int, default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler.", )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
                        )
    parser.add_argument(
        "--pre_compute_text_embeddings",
        action="store_true",
        help="Whether or not to pre-compute text embeddings. If text embeddings are pre-computed, the text encoder will not be kept in memory during training and will leave more GPU memory available for training the rest of the model. This is not compatible with `--train_text_encoder`.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def compute_text_embeddings(prompt, tokenizer, text_encoder):
    with torch.no_grad():
        text_inputs = tokenizer(prompt,
                                truncation=True,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                return_tensors="pt",
                                )
        prompt_embeds = encode_prompt(text_encoder, text_inputs.input_ids, text_encoder.device)
    return prompt_embeds


def encode_prompt(text_encoder, text_inputs_ids, device):
    with torch.no_grad():
        text_encoder.to(device)
        prompt_embeds = text_encoder(
            text_inputs_ids.to(device),
            attention_mask=None,
        )[0]
    return prompt_embeds


def predict_eps(
        alphas_cumprod,
        x_0_anomaly,
        x_0_normal,
        timesteps,
        noise
):
    x_0_anomaly = x_0_anomaly.to(torch.double)
    noise = noise.to(torch.double)
    x_0_normal = x_0_normal.to(torch.double)

    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = alphas_cumprod.to(device=x_0_anomaly.device, dtype=x_0_anomaly.dtype)

    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(x_0_anomaly.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(x_0_anomaly.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    eps = (sqrt_alpha_prod * x_0_anomaly + sqrt_one_minus_alpha_prod * noise - sqrt_alpha_prod * x_0_normal) / sqrt_one_minus_alpha_prod
    return eps.to(torch.float32)


def train_one_epoch(accelerator,
                    vae, text_encoder, unet, noise_scheduler,
                    train_dataloader, pre_encoder_hidden_states,
                    optimizer, lr_scheduler,
                    weight_dtype,
                    global_step, progress_bar,
                    args,
                    ):
    unet.train()
    while (True):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                with torch.no_grad():
                    instance_images = batch["anomaly_images"].to(dtype=weight_dtype)
                    supervise_images = batch["instance_images"].to(dtype=weight_dtype)

                    anomaly_input = vae.encode(instance_images).latent_dist.sample(torch.Generator(args.seed)) * vae.config.scaling_factor
                    supervise_input = vae.encode(supervise_images).latent_dist.sample(torch.Generator(args.seed)) * vae.config.scaling_factor

                    if pre_encoder_hidden_states is not None:
                        encoder_hidden_states = pre_encoder_hidden_states.to(accelerator.device)
                        encoder_hidden_states = encoder_hidden_states.repeat(anomaly_input.shape[0], 1, 1)
                    else:
                        encoder_hidden_states = encode_prompt(text_encoder, batch["instance_prompt_ids"].squeeze(), accelerator.device)

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (anomaly_input.shape[0],), device=accelerator.device).long()
                alpha = (timesteps / noise_scheduler.config.num_train_timesteps).reshape(timesteps.shape[0], 1, 1, 1)
                synthesis_features = alpha * anomaly_input + (1 - alpha) * supervise_input

                noise = torch.randn_like(anomaly_input)
                noisy_model_input = noise_scheduler.add_noise(synthesis_features, noise, timesteps)
                model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states).sample

                eps = predict_eps(noise_scheduler.alphas_cumprod, synthesis_features, supervise_input, timesteps, noise)
                loss = F.mse_loss(model_pred.float(), eps.float(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = (unet.parameters())
                    max_grad_norm = 1.0
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        # os.remove(f"{save_path}/optimizer.bin")
                        # os.remove(f"{save_path}/random_states_0.pkl")
                        # os.remove(f"{save_path}/scheduler.bin")
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                return loss, global_step


def test(dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step):
    print(f"test:{class_name}, checkpoint_step:{checkpoint_step}")

    with torch.no_grad():
        val_pipe.unet.load_state_dict(
            torch.load(args.output_dir + f"/checkpoint-{checkpoint_step}/pytorch_model.bin"))
        val_pipe.unet.to(dtype=weight_dtype)

        val_pipe.to(device)
        val_pipe.set_progress_bar_config(disable=True)
        val_pipe.unet.eval()
        val_pipe.vae.eval()
        val_pipe.text_encoder.eval()

        transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        preds = []
        masks = []
        scores = []
        labels = []

        for batch in val_dataloader:
            image_input = batch["instance_images"].to(device)
            anomaly_mask = batch["instance_masks"].to(device)
            object_mask = batch["object_mask"].to(device)

            reconstruct_images, step = reconstruction(val_pipe, weight_dtype, args, image_input, dino_frozen)

            image_input = torch.nn.functional.interpolate(image_input, size=args.dino_resolution, mode='bilinear', align_corners=True)
            reconstruct_images = torch.nn.functional.interpolate(reconstruct_images, size=args.dino_resolution, mode='bilinear', align_corners=True)
            anomaly_mask = torch.nn.functional.interpolate(anomaly_mask, size=args.dino_resolution, mode='bilinear', align_corners=True)
            object_mask = torch.nn.functional.interpolate(object_mask.unsqueeze(1), size=args.dino_resolution, mode='bilinear', align_corners=True).squeeze(1)

            image_input = transform(image_input)
            reconstruct_images = transform(reconstruct_images)

            _, patch_tokens_i = dino_model(image_input.to(dtype=weight_dtype))
            _, patch_tokens_r = dino_model(reconstruct_images.to(dtype=weight_dtype))

            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = patch_tokens_i[0][:, 1:, :].shape
            h = int(n ** 0.5)
            anomaly_maps1 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
            anomaly_maps2 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
            for idx in range(len(patch_tokens_i)):
                pi = patch_tokens_i[idx][:, 1:, :]
                pr = patch_tokens_r[idx][:, 1:, :]

                pi = pi / torch.norm(pi, p=2, dim=-1, keepdim=True)
                pr = pr / torch.norm(pr, p=2, dim=-1, keepdim=True)

                cos0 = torch.bmm(pi, pr.permute(0, 2, 1))

                anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True)
                anomaly_maps1 += anomaly_map1

                if class_name in ["transistor", "pcb1", "pcb4"]:
                    anomaly_map2, _ = torch.min(1 - cos0, dim=-2)
                    anomaly_map2 = F.interpolate(anomaly_map2.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True)
                    anomaly_maps2 += anomaly_map2

            if class_name in ["transistor", "pcb1", "pcb4"]:
                anomaly_maps1 = anomaly_maps1 + anomaly_maps2

            distance_map = torch.mean(torch.abs(image_input - reconstruct_images), dim=1, keepdim=True)
            anomaly_maps1 = anomaly_maps1 + args.v * torch.max(anomaly_maps1) / torch.max(distance_map) * distance_map

            anomaly_maps1 = gaussian_blur2d(anomaly_maps1, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))[:, 0]
            anomaly_maps = anomaly_maps1 * object_mask.to(device)
            score = torch.topk(torch.flatten(anomaly_maps, start_dim=1), 250)[0].mean(dim=1)

            masks.extend([m for m in anomaly_mask[:, 0, :, :].cpu().numpy()])
            preds.extend([a for a in anomaly_maps.cpu().numpy()])
            scores.extend([s for s in score.cpu().numpy()])
            labels.extend([l for l in batch["instance_label"].cpu().numpy()])

        scores = normalize(np.array(scores))
        labels = np.array(labels)
        preds = np.array(preds)
        masks = np.array(masks, dtype=np.int_)

        precisions_image, recalls_image, _ = precision_recall_curve(labels, scores)
        f1_scores_image = (2 * precisions_image * recalls_image) / (precisions_image + recalls_image)
        best_f1_scores_image = np.max(f1_scores_image[np.isfinite(f1_scores_image)])
        auroc_image = roc_auc_score(labels, scores)
        AP_image = average_precision_score(labels, scores)

        precisions_pixel, recalls_pixel, _ = precision_recall_curve(masks.ravel(), preds.ravel())
        f1_scores_pixel = (2 * precisions_pixel * recalls_pixel) / (precisions_pixel + recalls_pixel)
        best_f1_scores_pixel = np.max(f1_scores_pixel[np.isfinite(f1_scores_pixel)])
        auroc_pixel = roc_auc_score(masks.ravel(), preds.ravel())
        AP_pixel = average_precision_score(masks.ravel(), preds.ravel())

        pro = compute_pro(masks, preds)

        print(f"test-------- I-AUROC/I-AP/I-F1-max/P-AUROC/P-AP/P-F1-max/PRO:{round(auroc_image, 4)}/{round(AP_image, 4)}/{round(best_f1_scores_image, 4)}/"
              f"{round(auroc_pixel, 4)}/{round(AP_pixel, 4)}/{round(best_f1_scores_pixel, 4)}/{round(pro, 4)}-----")

        return round(auroc_image, 4)*100, round(AP_image, 4)*100, round(best_f1_scores_image, 4)*100, round(auroc_pixel, 4)*100, round(AP_pixel, 4)*100, round(best_f1_scores_pixel, 4)*100, round(pro, 4)*100

def visual(dino_model, dino_frozen, val_pipe, weight_dtype, val_dataloader, args, device, class_name, checkpoint_step):
    print(f"test:{class_name}, checkpoint_step:{checkpoint_step}")
    with torch.no_grad():

        val_pipe.unet.load_state_dict(torch.load(args.output_dir + f"/checkpoint-{checkpoint_step}/pytorch_model.bin"))

        val_pipe.unet.to(dtype=weight_dtype)
        val_pipe.to(device)
        val_pipe.set_progress_bar_config(disable=True)
        val_pipe.unet.eval()
        val_pipe.vae.eval()
        val_pipe.text_encoder.eval()

        transform = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / (2)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        path = f"{args.inference_step}NFE_{args.denoise_step}step_lcm/{args.output_dir.split('/')[-1]}_{args.input_threshold}"
        if os.path.exists(path) is not True:
            os.makedirs(path)

        preds = []
        masks = []
        scores = []
        labels = []
        rec_images = []
        sources = []
        steps = []

        for batch in val_dataloader:
            image_input = batch["instance_images"].to(device)
            anomaly_mask = batch["instance_masks"].to(device)
            object_mask = batch["object_mask"].to(device)

            reconstruct_images, step = reconstruction(val_pipe, weight_dtype, args, image_input, dino_frozen)

            image_input = torch.nn.functional.interpolate(image_input, size=args.dino_resolution, mode='bilinear', align_corners=True)
            reconstruct_images = torch.nn.functional.interpolate(reconstruct_images, size=args.dino_resolution, mode='bilinear', align_corners=True)
            anomaly_mask = torch.nn.functional.interpolate(anomaly_mask, size=args.dino_resolution, mode='bilinear', align_corners=True)
            object_mask = torch.nn.functional.interpolate(object_mask.unsqueeze(1), size=args.dino_resolution, mode='bilinear', align_corners=True).squeeze(1)

            image_input = transform(image_input)
            reconstruct_images = transform(reconstruct_images)

            _, patch_tokens_i = dino_model(image_input.to(dtype=weight_dtype))
            _, patch_tokens_r = dino_model(reconstruct_images.to(dtype=weight_dtype))

            sigma = 6
            kernel_size = 2 * int(4 * sigma + 0.5) + 1
            b, n, c = patch_tokens_i[0][:, 1:, :].shape
            h = int(n ** 0.5)
            anomaly_maps1 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
            anomaly_maps2 = torch.zeros((b, 1, args.dino_resolution, args.dino_resolution)).to(device)
            for idx in range(len(patch_tokens_i)):
                pi = patch_tokens_i[idx][:, 1:, :]
                pr = patch_tokens_r[idx][:, 1:, :]

                pi = pi / torch.norm(pi, p=2, dim=-1, keepdim=True)
                pr = pr / torch.norm(pr, p=2, dim=-1, keepdim=True)

                cos0 = torch.bmm(pi, pr.permute(0, 2, 1))

                anomaly_map1, _ = torch.min(1 - cos0, dim=-1)
                anomaly_map1 = F.interpolate(anomaly_map1.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True)
                anomaly_maps1 += anomaly_map1

                if class_name in ["transistor", "pcb1", "pcb4"]:
                    anomaly_map2, _ = torch.min(1 - cos0, dim=-2)
                    anomaly_map2 = F.interpolate(anomaly_map2.reshape(-1, 1, h, h), size=args.dino_resolution, mode='bilinear', align_corners=True)
                    anomaly_maps2 += anomaly_map2

            if class_name in ["transistor", "pcb1", "pcb4"]:
                anomaly_maps1 = anomaly_maps1 + anomaly_maps2

            distance_map = torch.mean(torch.abs(image_input - reconstruct_images), dim=1, keepdim=True)
            anomaly_maps1 = anomaly_maps1 + args.v * torch.max(anomaly_maps1) / torch.max(distance_map) * distance_map

            anomaly_maps1 = gaussian_blur2d(anomaly_maps1, kernel_size=(kernel_size, kernel_size),
                                            sigma=(sigma, sigma))[:, 0]
            anomaly_maps = anomaly_maps1 * object_mask.to(device)
            score = torch.topk(torch.flatten(anomaly_maps, start_dim=1), 250)[0].mean(dim=1)

            masks.extend([m for m in anomaly_mask[:, 0, :, :].cpu().numpy()])
            preds.extend([a for a in anomaly_maps.cpu().numpy()])
            scores.extend([s for s in score.cpu().numpy()])
            labels.extend([l for l in batch["instance_label"].cpu().numpy()])
            steps.extend([t.cpu().numpy() for t in step])

            for i in range(len(image_input)):
                source = image_input[i].cpu().permute(1, 2, 0).numpy() * (0.229, 0.224, 0.225) + (
                    0.485, 0.456, 0.406)
                reconstruct_image = reconstruct_images[i].cpu().permute(1, 2, 0).numpy() * (
                    0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
                rec_images.append(reconstruct_image)
                sources.append(source)

        scores = normalize(np.array(scores))
        labels = np.array(labels)
        preds = np.array(preds)
        masks = np.array(masks, dteype=np.int_)

        precisions_image, recalls_image, _ = precision_recall_curve(labels, scores)
        f1_scores_image = (2 * precisions_image * recalls_image) / (precisions_image + recalls_image)
        best_f1_scores_image = np.max(f1_scores_image[np.isfinite(f1_scores_image)])
        auroc_image = roc_auc_score(labels, scores)
        AP_image = average_precision_score(labels, scores)

        precisions_pixel, recalls_pixel, _ = precision_recall_curve(masks.ravel(), preds.ravel())
        f1_scores_pixel = (2 * precisions_pixel * recalls_pixel) / (precisions_pixel + recalls_pixel)
        best_f1_scores_pixel = np.max(f1_scores_pixel[np.isfinite(f1_scores_pixel)])
        auroc_pixel = roc_auc_score(masks.ravel(), preds.ravel())
        AP_pixel = average_precision_score(masks.ravel(), preds.ravel())

        pro = compute_pro(masks, preds)

        print(
            f"test-------- I-AUROC/I-AP/I-F1-max/P-AUROC/P-AP/P-F1-max/PRO:{round(auroc_image, 4)}/{round(AP_image, 4)}/{round(best_f1_scores_image, 4)}/"
            f"{round(auroc_pixel, 4)}/{round(AP_pixel, 4)}/{round(best_f1_scores_pixel, 4)}/{round(pro, 4)}-----"
        )

        heat = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))

        rec_images = np.clip(rec_images, 0.0, 1.0)
        for i, batch in enumerate(val_dataloader.dataset):
            heat1 = utilize.apply_ad_scoremap(sources[i] * 255, heat[i])
            mask = masks[i][:, :, None].repeat(3, axis=2) * 255
            img = np.hstack([sources[i][:, :, ::-1] * 255, rec_images[i][:, :, ::-1] * 255, heat1[:, :, ::-1], mask])
            name = batch["instance_path"].split("/")
            name = f"{path}/{format(scores[i], '.4f')}_{steps[i]}_{name[-2]}_{name[-1]}"
            cv2.imwrite(name, img.astype(np.uint8))

        return round(auroc_image, 4) * 100, round(AP_image, 4) * 100, round(best_f1_scores_image, 4) * 100, \
            round(auroc_pixel, 4) * 100, round(AP_pixel, 4) * 100, round(best_f1_scores_pixel, 4) * 100, round(pro, 4) * 100

def load_vae(vae, class_name):
    if class_name in ["pcb1", "pcb2", "pcb3", "pcb4", "pcb5", "pcb6", "pcb7"]:
        vae_path_state = {
            "pcb1": f"/hdd/yaohang/DiffAD-main/logs/2024-04-06T18-05-16_kl_pcb1/checkpoints/epoch=000069.ckpt",
            "pcb2": f"/hdd/yaohang/DiffAD-main/logs/2024-04-07T07-35-18_kl_pcb2/checkpoints/epoch=000053.ckpt",
            "pcb3": f"/hdd/yaohang/DiffAD-main/logs/2024-04-08T08-26-24_kl_pcb3/checkpoints/epoch=000096.ckpt",
            "pcb4": f"/hdd/yaohang/DiffAD-main/logs/2024-04-09T06-58-50_kl_pcb4/checkpoints/epoch=000058.ckpt",
            "pcb5": f"/hdd/yaohang/DiffAD-main/logs/2024-05-21T07-05-12_kl_pcb5/checkpoints/last.ckpt",
            "pcb6": f"/hdd/yaohang/DiffAD-main/logs/2024-06-01T15-26-49_kl_pcb6/checkpoints/last.ckpt",
            "pcb7": f"/hdd/yaohang/DiffAD-main/logs/2024-05-23T09-14-13_kl_pcb7/checkpoints/last.ckpt",
        }
        vae_path = vae_path_state[class_name]
        sd = torch.load(vae_path)["state_dict"]
        print(f"load vae in test :{vae_path}")

        keys = list(sd.keys())
        for k in keys:
            if "loss" in k:
                del sd[k]
        vae.load_state_dict(sd)
        return vae
    else:
        return vae


def load_test_model(args, weight_dtype):
    dino_model = get_vit_encoder(vit_arch="vit_base", vit_model="dino", vit_patch_size=8, enc_type_feats=None).to(device, dtype=weight_dtype)
    dino_model.eval()

    val_pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
        torch_dtype=weight_dtype,
    )
    return dino_model, val_pipe


def main(args, class_name):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers("GLAD", config=tracker_config)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )
    text_encoder, vae, unet = model(args)

    vae.to(accelerator.device, dtype=weight_dtype)
    vae = load_vae(vae, class_name)

    text_encoder.to(accelerator.device, dtype=weight_dtype)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae.eval()
    text_encoder.eval()

    if args.pre_compute_text_embeddings:
        pre_encoder_hidden_states = compute_text_embeddings(args.instance_prompt, tokenizer, text_encoder)
    else:
        pre_encoder_hidden_states = None

    optimizer_class = bnb.optim.AdamW8bit
    params_to_optimize = (unet.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    train_dataset = MVTecDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_name=class_name,
        tokenizer=tokenizer,
        resize=args.resolution,
        img_size=args.resolution,
        anomaly_path=args.anomaly_data_dir,
        train=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_epochs = round(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Class name = {class_name}")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  save model into {args.output_dir}")
    global_step = 0

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    loss, global_step = train_one_epoch(accelerator,
                                        vae, text_encoder, unet, noise_scheduler,
                                        train_dataloader, pre_encoder_hidden_states,
                                        optimizer, lr_scheduler,
                                        weight_dtype,
                                        global_step, progress_bar,
                                        args)
    logger.info(f"train--------train loss:{loss}-----")
    del vae
    del text_encoder
    del unet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda")
    dino_model, val_pipe = load_test_model(args, torch.float16)
    dino_frozen = copy.deepcopy(dino_model)

    if args.class_name == "all":
        if 'MVTec-AD' in args.instance_data_dir:
            CLSNAMES = {
                'carpet': [750, 0.32, 0, 350, 2000, 0],
                'grid': [750, 0.47, 0, 350, 3000, 0],
                'leather': [750, 0.35, 0, 350, 2500, 0],
                'tile': [750, 0.35, 0, 350, 500, 0],
                'wood': [750, 0.37, 0, 350, 1000, 0],
                'bottle': [750, 0.32, 0, 350, 2500, 0],
                'cable': [750, 0.4, 0, 350, 1500, 0],
                "capsule": [600, 0.4, 0, 350, 2000, 0],
                'hazelnut': [750, 0.5, 0, 350, 3500, 0],
                'metal_nut': [750, 0.4, 0, 350, 2500, 0],
                'pill': [750, 0.35, 0, 350, 1500, 0], 
                'screw': [750, 0.32, 0, 350, 3000, 0],
                'toothbrush': [750, 0.5, 0, 350, 500, 0],
                'transistor': [850, 0.5, 0, 350, 2500, 0], 
                'zipper': [750, 0.35, 0, 350, 1000, 0],
            }
            args.save_path = None
            args.dino_resolution = 512
            args.test_batch_size = 2
            args.inference_step = 25
            args.output_dir = "4000step_bs32_eps_anomaly2"
        elif 'MPDD' in args.instance_data_dir:
            CLSNAMES = {
                'bracket_black':[500, 0.35, 0, 350, 2500, 0],
                'bracket_brown':[500, 0.35, 0, 350, 7000, 0],
                'bracket_white':[450, 0.35, 0, 200, 1000, 0],
                'connector':[500, 0.35, 0, 350, 1000, 0],
                'metal_plate':[500, 0.35, 0, 350, 2500, 0],
                'tubes':[500, 0.1, 0, 350, 500, 0],
            }
            args.save_path = None
            args.dino_resolution = 512
            args.test_batch_size = 2
            args.inference_step = 25
            args.output_dir = "4000step_bs32_eps_anomaly2"
        elif 'VisA' in args.instance_data_dir:
            CLSNAMES = {
                'candle':  [450, 0.45, 7, 200, 4000, 1],
                'capsules':[450, 0.4, 7, 200, 4000, 4],
                'cashew':  [450, 0.4, 1, 200, 4000, 7],
                'chewinggum':[450, 0.45, 1, 200, 4000, 7],
                'fryum':[450, 0.35, 2, 200, 4000, 1],
                'macaroni1':[450, 0.45, 7, 200, 4000, 1],
                'macaroni2':[450, 0.45, 7, 200, 4000, 8],
                'pcb1':[450, 0.3, 1, 200, 4000, 7],
                'pcb2':[450, 0.3, 2, 200, 4000, 6],
                'pcb3':[450, 0.3, 2, 200, 4000, 9],
                'pcb4':[450, 0.3, 2, 200, 4000, 0],
                'pipe_fryum':[450, 0.45, 2, 200, 4000, 2],
            }
            args.save_path = "model/visa_dino"
            args.dino_resolution = 256
            args.test_batch_size = 32
            args.inference_step = 15
            args.output_dir = "8000step_bs32_eps_anomaly2"

        elif 'PCBBank' in args.instance_data_dir:
            CLSNAMES = {
                'pcb1': [450, 0.3, 1, 200, 4000, 7],
                'pcb2': [450, 0.3, 2, 200, 4000, 6],
                'pcb3': [450, 0.3, 2, 200, 4000, 9],
                'pcb4': [450, 0.3, 2, 200, 4000, 0],
                'pcb5': [450, 0.4, 2, 200, 4000, 7],
                'pcb6': [450, 0.45, 1, 200, 4000, 14],
                'pcb7': [450, 0.3, 2, 200, 4000, 1],
            }
            args.save_path = "model/pcbbank_dino"
            args.dino_resolution = 256
            args.test_batch_size = 32
            args.inference_step = 15
            args.output_dir = "8000step_bs32_eps_anomaly2"
        else:
            raise ValueError("None dataset")

        performances = [[], [], [], [], [], [], []]
        output_dir = args.output_dir
        for class_name in CLSNAMES:

            val_pipe.vae = load_vae(val_pipe.vae, class_name)
            fix_seeds(args.seed)
            args.output_dir = os.path.join("model", class_name + "_" + output_dir + f"_{args.seed}")

            if args.train:
                print(class_name, args)
                main(args, class_name)
            else:
                val_dataset = MVTecDataset(
                    instance_data_root=args.instance_data_dir,
                    instance_prompt=args.instance_prompt,
                    class_name=class_name,
                    tokenizer=None,
                    resize=args.resolution,
                    img_size=args.resolution,
                    train=False
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.test_batch_size,
                    num_workers=args.dataloader_num_workers,
                    shuffle=False,
                    pin_memory=True,
                )

                args.denoise_step = CLSNAMES[class_name][0]
                args.input_threshold = CLSNAMES[class_name][1]
                args.v = CLSNAMES[class_name][2]
                args.min_step = CLSNAMES[class_name][3]
                checkpoint_step = CLSNAMES[class_name][4]
                dino_epoch = CLSNAMES[class_name][5]
                print(class_name, args)
                print(f"Test checkpoint step {checkpoint_step}", time.asctime())
                print(f"denoise_step:{args.denoise_step}, input_threshold:{args.input_threshold}, min_step: {args.min_step}")

                if args.save_path:
                    if class_name in ["cashew", "macaroni2"]:
                        lmd = 0.0
                    else:
                        lmd = 0.01
                    text = f"4mlp_{args.dino_resolution}_{args.min_step}_bs16_0.0003_15_no_grad2_lmd{lmd}"

                    dino_model.load_state_dict(
                        torch.load(f"{args.save_path}/{class_name}_{text}/epoch{dino_epoch}.pth")
                    )

                results = test(
                    dino_model, dino_frozen, val_pipe, torch.float16, val_dataloader, args, device,
                    class_name,
                    checkpoint_step
                    )
                for i, result in enumerate(results):
                    performances[i].append(result)
                    
        performances = np.array(performances).T
        print(np.mean(performances, axis=0))
