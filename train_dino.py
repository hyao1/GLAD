import creat_model
from utilize import utilize
import torch
import argparse
from torch import optim
from dataset.dataset import MVTecDataset
from torch.utils.data import DataLoader
from utilize.utilize import reconstruction
import os
import copy
from torchvision import transforms, utils
import time
from pipeline import StableDiffusionPipeline
from ddim_scheduling import DDIMScheduler
from creat_model import get_vit_encoder
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import logging
from accelerate.logging import get_logger

logger = get_logger(__name__)

pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--instance_data_dir", default=None, type=str)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--inference_step", default=25, type=int)
    parser.add_argument("--test_batch_size", default=2, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--lmd", default=0.0, type=float)
    parser.add_argument("--resolution", default=300, type=int)
    parser.add_argument("--class_name", default="", type=str)
    parser.add_argument("--denoise_step", default=300)
    parser.add_argument("--min_step", default=200, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataloader_num_workers", default=0, type=int)
    parser.add_argument("--instance_prompt", default='a photo of sks', type=str)
    parser.add_argument("--mixed_precision", default='no', type=str)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    return args


def loss_fucntion(i, r, fi=None, fr=None, lmd=0.0):
    cos_loss = torch.nn.CosineSimilarity()
    loss1 = 0
    loss2 = 0
    loss3 = 0
    if lmd > 0.0:
        for item in range(len(i)):
            loss1 += torch.mean(1 - cos_loss(i[item].view(i[item].shape[0], -1), r[item].view(r[item].shape[0], -1)))
            loss2 += torch.mean(
                1 - cos_loss(i[item].view(i[item].shape[0], -1), fi[item].view(fi[item].shape[0], -1))) * lmd
            loss3 += torch.mean(
                1 - cos_loss(r[item].view(r[item].shape[0], -1), fr[item].view(fr[item].shape[0], -1))) * lmd
        loss = loss1 + loss2 + loss3
        return loss
    elif args.lmd <= 0.0:
        for item in range(len(i)):
            loss1 += torch.mean(1 - cos_loss(i[item].view(i[item].shape[0], -1), r[item].view(r[item].shape[0], -1)))
        return loss1


def main(args, device):
    logging_dir = os.path.join(f"{args.save_path}/{args.class_name}_{args.text}")
    accelerator_project_config = ProjectConfiguration(project_dir=f"{args.save_path}/{args.class_name}_{args.text}",
                                                      logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="all",
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(f"{args.save_path}/{args.class_name}_{args.text}", exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(args)
    logger.info(accelerator.state, main_process_only=False)

    weight_dtype = torch.float32

    dino_model = get_vit_encoder(vit_arch="vit_base", vit_model="dino", vit_patch_size=8, enc_type_feats=None)
    for name, param in dino_model.named_parameters():
        if "blocks.2.mlp" in name or "blocks.5.mlp" in name or "blocks.8.mlp" in name or "blocks.11.mlp" in name:
            param.requires_grad = True
    dino_model = dino_model.to(accelerator.device, dtype=weight_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler"),
        torch_dtype=torch.float16,
    )
    pipe.unet.load_state_dict(
        torch.load(args.output_dir + f"/checkpoint-{args.checkpoint_step}/pytorch_model.bin")
    )
    pipe.unet.to(dtype=torch.float16)

    pipe = pipe.to(accelerator.device)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)

    if args.class_name in ["pcb1", "pcb2", "pcb3", "pcb4", "pcb5", "pcb6", "pcb7"]:
        vae_path_state = {
            "pcb1": f"/hdd/yaohang/DiffAD-main/logs/2024-04-06T18-05-16_kl_pcb1/checkpoints/epoch=000069.ckpt",
            "pcb2": f"/hdd/yaohang/DiffAD-main/logs/2024-04-07T07-35-18_kl_pcb2/checkpoints/epoch=000053.ckpt",
            "pcb3": f"/hdd/yaohang/DiffAD-main/logs/2024-04-08T08-26-24_kl_pcb3/checkpoints/epoch=000096.ckpt",
            "pcb4": f"/hdd/yaohang/DiffAD-main/logs/2024-04-09T06-58-50_kl_pcb4/checkpoints/epoch=000058.ckpt",
            "pcb5": f"/hdd/yaohang/DiffAD-main/logs/2024-05-21T07-05-12_kl_pcb5/checkpoints/last.ckpt",
            "pcb6": f"/hdd/yaohang/DiffAD-main/logs/2024-06-01T15-26-49_kl_pcb6/checkpoints/last.ckpt",
            "pcb7": f"/hdd/yaohang/DiffAD-main/logs/2024-05-23T09-14-13_kl_pcb7/checkpoints/last.ckpt",
        }
        vae_path = vae_path_state[args.class_name]

        sd = torch.load(vae_path)["state_dict"]
        logger.info(f"load vae in :{vae_path}")

        keys = list(sd.keys())
        for k in keys:
            if "loss" in k:
                del sd[k]
        pipe.vae.load_state_dict(sd)

    optimizer = torch.optim.Adam([
        {"params": filter(lambda p: p.requires_grad, dino_model.parameters()), "lr": args.lr}
    ])

    train_dataset = MVTecDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_name=args.class_name,
        tokenizer=None,
        resize=512,
        img_size=512,
        train=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    (
        dino_model,
        optimizer,
        train_dataloader,
    ) = accelerator.prepare(
        dino_model, optimizer, train_dataloader
    )

    sigma = 6
    kernel_size = 2 * int(4 * sigma + 0.5) + 1
    transform = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / (2)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.lmd > 0:
        frozen_dino = copy.deepcopy(dino_model)
        frozen_dino.requires_grad_(False)

    for epoch in range(args.epochs):

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(dino_model).state_dict(),
                       f"{args.save_path}/{args.class_name}_{args.text}/epoch{epoch}.pth")

        logger.info(
            f"epoch {epoch} begining: save model to {args.save_path}/{args.class_name}_{args.text}/epoch{epoch}.pth")
        logger.info(f"time:{time.time()}, mem:{torch.cuda.memory_reserved()}")
        dino_model.train()
        for step, sample in enumerate(train_dataloader):

            with accelerator.accumulate(dino_model):
                with torch.no_grad():
                    image = sample["instance_images"].to(accelerator.device)

                    reconstruct_images, _ = reconstruction(pipe, torch.float16, args, image, dino_model=None)

                    reconstruct_images = reconstruct_images.to(dtype=weight_dtype)
                    image = image.to(dtype=weight_dtype)

                    image = torch.nn.functional.interpolate(image, size=args.resolution, mode='bilinear',
                                                            align_corners=True)
                    reconstruct_images = torch.nn.functional.interpolate(reconstruct_images, size=args.resolution,
                                                                         mode='bilinear', align_corners=True)

                    image = transform(image).detach()
                    reconstruct_images = transform(reconstruct_images).detach()

                _, patch_tokens_i = dino_model(image)
                _, patch_tokens_r = dino_model(reconstruct_images)

                if args.lmd > 0.0:
                    _, patch_tokens_i_f = frozen_dino(image)
                    _, patch_tokens_r_f = frozen_dino(reconstruct_images)

                    loss = loss_fucntion(patch_tokens_i, patch_tokens_r, patch_tokens_i_f, patch_tokens_r_f, args.lmd)
                elif args.lmd <= 0.0:
                    loss = loss_fucntion(patch_tokens_i, patch_tokens_r)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
        logger.info(f"train-------- loss:{loss}")


if __name__ == "__main__":
    args = parse_args()
    args.instance_data_dir = f"/hdd/Datasets/{args.dataset}"

    MVTecAD_class_name = {
        'carpet': [2000, 350, 0.1],
        'grid': [3000, 350, 0.1],
        'leather': [2500, 350, 0.1],
        'tile': [500, 350, 0.1],
        'wood': [1000, 350, 0.1],
        'bottle': [2500, 350, 0.1],
        'cable': [1500, 350, 0.1],
        'capsule': [2000, 350, 0.1],
        'hazelnut': [3500, 350, 0.1],
        'metal_nut': [2500, 350, 0.1],
        'pill': [1500, 350, 0.1],
        'screw': [3000, 350, 0.1],
        'toothbrush': [500, 350, 0.1],
        'transistor': [2500, 350, 0.1],
        'zipper': [1000, 350, 0.1]
    }

    VisA_class_name = {
        'candle': [4000, 200, 0.01],
        'capsules': [4000, 200, 0.01],
        'cashew': [4000, 200, 0.0],
        'chewinggum': [4000, 200, 0.01],
        'fryum': [4000, 200, 0.01],
        'macaroni1': [4000, 200, 0.01],
        'macaroni2': [4000, 200, 0.0],
        'pcb1': [4000, 200, 0.0],
        'pcb2': [4000, 200, 0.01],
        'pcb3': [4000, 200, 0.01],
        'pcb4': [4000, 200, 0.01],
        'pipe_fryum': [4000, 200, 0.01],
    }

    PCBBank_class_name = {
        'pcb1': [4000, 200, 0.01],
        'pcb2': [4000, 200, 0.01],
        'pcb3': [4000, 200, 0.01],
        'pcb4': [4000, 200, 0.01],
        'pcb5': [4000, 200, 0.01],
        'pcb6': [4000, 200, 0.01],
        'pcb7': [4000, 200, 0.01],
    }
    if args.dataset == "VisA":
        class_name = VisA_class_name
        args.save_path = "model/visa_dino"
        total_checkpoints = "8000"
        args.denoise_step = 200
        args.resolution = 256
        args.train_batch_size = 16
        args.gradient_accumulation_steps = 2

    elif args.dataset == "PCBBank":
        class_name = PCBBank_class_name
        args.save_path = "model/pcbbank_dino"
        total_checkpoints = "8000"
        args.denoise_step = 200
        args.resolution = 256
        args.train_batch_size = 16
        args.gradient_accumulation_steps = 2

    for args.class_name in class_name:
        args.checkpoint_step = class_name[args.class_name][0]
        args.lmd = class_name[args.class_name][2]

        args.output_dir = f"model/{args.class_name}_{total_checkpoints}step_bs32_eps_anomaly_0"
        args.lr = 3e-4
        args.inference_step = 15

        args.text = f"4mlp_{args.resolution}_{args.denoise_step}_bs{args.train_batch_size}_{args.lr}_{args.inference_step}_{args.mixed_precision}_grad{args.gradient_accumulation_steps}_lmd{args.lmd}"
        device = torch.device("cuda")
        utilize.fix_seeds(args.seed)
        main(args, device)
