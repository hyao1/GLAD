from transformers import CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
import torch
import dino.vision_transformer as vits

def model(args):
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    vae.eval()
    text_encoder.eval()
    return text_encoder, vae, unet


def get_vit_encoder(vit_arch, vit_model, vit_patch_size, enc_type_feats=None):
    if vit_model == "dino":
        if vit_arch == "vit_small" and vit_patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            initial_dim = 384
        elif vit_arch == "vit_small" and vit_patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            initial_dim = 384
        elif vit_arch == "vit_base" and vit_patch_size == 16:
            if vit_model == "clip":
                url = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
            elif vit_model == "dino":
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            initial_dim = 768
        elif vit_arch == "vit_base" and vit_patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            initial_dim = 768

        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/" + url
        )
        vit_encoder = vits.__dict__[vit_arch](patch_size=vit_patch_size, num_classes=0, extraction=[3, 6, 9, 12])
        vit_encoder.load_state_dict(state_dict, strict=True)

    elif vit_model == "dino-v2":
        if vit_model == "dino-v2" and vit_arch == "vit_base" and vit_patch_size == 14:
            # url = "dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
            # initial_dim = 768
            vit_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        elif vit_model == "dino-v2" and vit_arch == "vit_large" and vit_patch_size == 14:
            # url = "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
            # initial_dim = 768
            vit_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')

        # state_dict = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/dinov2/" + url
        # )
        # vit_encoder = vits2.__dict__[vit_arch](patch_size=vit_patch_size)

    for p in vit_encoder.parameters():
        p.requires_grad = False

    return vit_encoder