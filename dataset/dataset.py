from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL.ImageOps import exif_transpose
import os
import numpy as np
import torch
from einops import rearrange
import cv2
import glob
import imgaug.augmenters as iaa
from dataset.perlin_noise import rand_perlin_2d_np
import scipy.ndimage as ndimage


class MVTecDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root,
            instance_prompt,
            tokenizer,
            resize=512,
            img_size=512,
            train=True,
            class_name="",
            encoder_hidden_states=None,
            anomaly_path='/hdd/Datasets/dtd'
    ):

        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.resize = resize
        self.img_size = img_size
        self.instance_data_root = instance_data_root
        self.anomaly_path = anomaly_path
        self.class_name = class_name
        self.type = 'train' if train else 'test'

        self.instance_images_paths, self.instance_masks_paths = self.get_data_single_class(self.instance_data_root)

        self.num_instance_images = len(self.instance_images_paths)
        self.num_mask_images = len(self.instance_masks_paths)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        self.structure_grid_size = 8

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.transforms_resize = transforms.Resize((self.resize, self.resize),
                                                   interpolation=transforms.InterpolationMode.BILINEAR)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        instance_images_path = self.instance_images_paths[index]
        example["instance_path"] = instance_images_path
        instance_image = Image.open(instance_images_path)
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_mask_path = self.instance_masks_paths[index]
        if instance_mask_path:
            instance_mask = Image.open(instance_mask_path).convert('L')
            instance_mask = exif_transpose(instance_mask)
            example["instance_label"] = 1
        else:
            instance_mask = Image.fromarray(np.zeros([*instance_image.size]))
            example["instance_label"] = 0
        example["instance_images"], example["instance_masks"] = self.transformer_compose(instance_image, instance_mask)

        if self.encoder_hidden_states:
            example["encoder_hidden_states"] = self.encoder_hidden_states
        else:
            if self.tokenizer:
                text_inputs = self.tokenizer(
                    self.instance_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )
                example["instance_prompt_ids"] = text_inputs.input_ids
                example["instance_attention_mask"] = text_inputs.attention_mask

        instance_image_resize = self.transforms_resize(instance_image)

        if self.type == 'train':
            anomaly_image, anomaly_mask, anomaly_label = self.generate_anomaly(np.array(instance_image_resize))
            example["anomaly_images"], example["anomaly_masks"] = self.transformer_compose(anomaly_image, anomaly_mask)
            example["instance_label"] = anomaly_label
        else: 
            if self.class_name in ["screw", "capsule", "bracket_black", "bracket_brown"]:
                mode = 1
            elif self.class_name in ["bracket_white"]:
                mode = 2
            else:
                mode = 3

            foreground_mask = self.generate_target_foreground_mask(np.array(instance_image_resize), mode=mode).astype(np.float32)
            sigma = 6.0
            object_mask = ndimage.gaussian_filter(foreground_mask, sigma=sigma)
            object_mask = np.where(object_mask > 0, 1.0, 0.0)
            object_mask = ndimage.gaussian_filter(object_mask, sigma=sigma)
            example["object_mask"] = object_mask

        return example

    def generate_anomaly(self, image):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        anomaly_source_paths = sorted(glob.glob(f"{self.anomaly_path}/images" + "/*/*.jpg"))
        anomaly_source_image = self.anomaly_source(image, anomaly_source_paths, aug)

        perlin_scale = 6
        min_perlin_scale = 0
        threshold = 0.3
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2).astype(np.float32)

        if self.class_name in ["screw", "bottle", "capsule", "zipper", "bracket_black", "bracket_brown",
                                "metal_plate"]:
            mode = 1
        elif self.class_name in ["hazelnut", "pill", "metal_nut", "toothbrush", "candle", "cashew", "chewinggum",
                                "fryum", "macaroni1", "macaroni2", "pipe_fryum", "bracket_white"]:
            mode = 2
        elif self.class_name in ["tile", "grid", "cable", "carpet", "leather", "wood", "transistor", "capsules",
                                "pcb1", "pcb2", "pcb3", "pcb4", "connector", "tubes"]:
            mode = 3
        else:
            mode = 3

        foreground_mask = self.generate_target_foreground_mask(image, mode=mode).astype(np.float32)
        perlin_thr = np.expand_dims(foreground_mask, axis=2) * perlin_thr

        anomaly_source_thr = anomaly_source_image * perlin_thr
        beta = torch.rand(1).numpy()[0] * 0.8
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * anomaly_source_thr + beta * image * perlin_thr

        anomaly = torch.rand(1).numpy()[0]
        if anomaly > 0.5:
            augmented_image = augmented_image.astype(np.uint8)
            msk = (perlin_thr * 255).astype(np.uint8).squeeze()
            has_anomaly = 0 if np.sum(msk) == 0 else 1

            return Image.fromarray(augmented_image), Image.fromarray(msk), has_anomaly
        else:
            mask = np.zeros_like(perlin_thr).astype(np.uint8).squeeze()
            return Image.fromarray(image.astype(np.uint8)), Image.fromarray(mask), 0

    def generate_target_foreground_mask(self, img: np.ndarray, mode) -> np.ndarray:
        # Converting RGB into grayscale

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if mode == 1:  # USING THIS FOR NOT WHITE BACKGROUND
            _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_background_mask = target_background_mask.astype(bool).astype(int)
            # Inverting mask for foreground mask
            target_foreground_mask = -(target_background_mask - 1)

        elif mode == 2:  # USING THIS FOR DARK BACKGROUND
            _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            target_background_mask = target_background_mask.astype(bool).astype(int)
            target_foreground_mask = target_background_mask

        elif mode == 3:
            target_foreground_mask = np.ones(img_gray.shape)
        else:
            target_foreground_mask = None

        return target_foreground_mask

    def anomaly_source(self, img, anomaly_path_list, aug):
        p = np.random.uniform()
        if p < 0.5:
            idx = np.random.choice(len(anomaly_path_list))
            texture_source_img = cv2.imread(anomaly_path_list[idx])
            texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
            anomaly_source_img = cv2.resize(texture_source_img, (self.resize, self.resize))
            anomaly_source_img = aug(image=anomaly_source_img)
        else:
            structure_source_img = aug(image=img)
            assert self.resize % self.structure_grid_size == 0, 'structure should be devided by grid size accurately'
            grid_w = self.resize // self.structure_grid_size
            grid_h = self.resize // self.structure_grid_size

            structure_source_img = rearrange(
                tensor=structure_source_img,
                pattern='(h gh) (w gw) c -> (h w) gw gh c',
                gw=grid_w,
                gh=grid_h
            )
            disordered_idx = np.arange(structure_source_img.shape[0])
            np.random.shuffle(disordered_idx)

            anomaly_source_img = rearrange(
                tensor=structure_source_img[disordered_idx],
                pattern='(h w) gw gh c -> (h gh) (w gw) c',
                h=self.structure_grid_size,
                w=self.structure_grid_size
            ).astype(np.float32)

        return anomaly_source_img

    def transformer_compose(self, image, mask):
        transforms_to_tensor = transforms.ToTensor()
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transforms_normalize = transforms.Normalize(mean, std)

        image = self.transforms_resize(image)
        image = transforms_to_tensor(image)
        image = transforms_normalize(image)

        if mask:
            mask = self.transforms_resize(mask)
            mask = transforms_to_tensor(mask)
        return image, mask

    def get_data_single_class(self, instance_data_root):
        image_list = []
        mask_list = []
        object_fold = self.class_name
        mask_fold = '' if self.type == 'train' else 'ground_truth'

        foldnames = os.listdir(os.path.join(instance_data_root, self.class_name, self.type))
        for image_fold in foldnames:
            filenames = os.listdir(os.path.join(instance_data_root, self.class_name, self.type, image_fold))
            for img in filenames:
                image_list.append(os.path.join(instance_data_root, self.class_name, self.type, image_fold, img))
                if image_fold == 'good':
                    mask_list.append(None)
                else:
                    if 'VisA' in instance_data_root or 'PCBBank' in instance_data_root:
                        mask_name = img.split('.')[0] + ".png"
                    else:
                        mask_name = img.split('.')[0] + "_mask." + img.split('.')[1]
                    mask_list.append(os.path.join(instance_data_root, object_fold, mask_fold, image_fold, mask_name))
        return image_list, mask_list
