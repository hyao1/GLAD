import torch
import cv2
import numpy as np
import random
import os
from skimage import measure
import pandas as pd
from statistics import mean
from sklearn.metrics import auc


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float_)
    scoremap = (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min())
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def fix_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_pro(masks, amaps, num_th=200):

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th
    binary_amaps = np.zeros_like(amaps)
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks , binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": mean(pros), "fpr": fpr, "threshold": th}, index=[0])], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def reconstruction(pipe, weight_dtype, args, batch_image, dino_model=None):
    with torch.no_grad():
        latents = pipe.vae.encode(
            batch_image.to(dtype=weight_dtype)).latent_dist.sample() * pipe.vae.config.scaling_factor
        if 'MVTec-AD' in args.instance_data_dir or 'MPDD' in args.instance_data_dir: # it is not important to use which timesteps. This code is just for reimplementation
            timesteps = torch.tensor([args.denoise_step], device=latents.device).long()  # for mvtec and mpdd
        else:
            timesteps = torch.randint(args.denoise_step - 1, args.denoise_step, (latents.shape[0],), device=latents.device).long() # for visa and pcbbank

        noise = torch.randn_like(latents).to(latents.device)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
        pipe.scheduler.config.num_train_timesteps = args.denoise_step

        if dino_model:
            reconstruct_images, step, threshold = pipe(args.instance_prompt,
                                                       num_images_per_prompt=batch_image.shape[0],
                                                       num_inference_steps=args.inference_step, guidance_scale=9,
                                                       latents=noisy_latents,
                                                       image_latents=latents,
                                                       dino_model=dino_model,
                                                       noise=noise,
                                                       denoise_step=args.denoise_step,
                                                       input_threshold=args.input_threshold,
                                                       min_step=args.min_step
                                                       )
            return reconstruct_images, step

        else:
            reconstruct_images, _, threshold = pipe(args.instance_prompt,
                                                    num_images_per_prompt=batch_image.shape[0],
                                                    num_inference_steps=args.inference_step, guidance_scale=9,
                                                    latents=noisy_latents)

            return reconstruct_images, torch.tensor([args.denoise_step] * reconstruct_images.shape[0])