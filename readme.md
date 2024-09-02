# GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection.
[ECCV2024]The official code of ["GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection"](https://arxiv.org/abs/2406.07487). 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/glad-towards-better-reconstruction-with/anomaly-detection-on-visa)](https://paperswithcode.com/sota/anomaly-detection-on-visa?p=glad-towards-better-reconstruction-with)

![image](https://github.com/hyao1/GLAD/assets/52654892/62a8d52d-72ab-4bda-8fb4-41b5d8e0a044)

## News
* [07/13/2024] Support for mixed_precision training.
* [07/13/2024] Release the [PCB-Bank](https://github.com/SSRheart/PCB-Bank) dataset we integrated.
  
## Requirements
This repository is implemented and tested on Python 3.10 and PyTorch 2.0.1.
To install requirements:

```setup
pip install -r requirements.txt
```

## Models Trained by Us
Models (VAE, Unet, DINO) trained by us are here: [OneDrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/23b903042_stu_hit_edu_cn/Etg1bdDSnOZBt7AydlkCzMUBYKxgmM_9tB-g5M70PJhAVQ).

## Training and Evaluation of the Model for Single-class
First, you should download the pretrained stable diffusion model from [pretrained model](https://huggingface.co/CompVis/stable-diffusion-v1-4), and datasets. (In addition, [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset is required for anomaly synthesis). If you can not download pretrained stable diffusion model, we provided it in our [OneDrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/23b903042_stu_hit_edu_cn/Etg1bdDSnOZBt7AydlkCzMUBYKxgmM_9tB-g5M70PJhAVQ).

To train the UNet of stable diffusion, modify the settings in the train.sh and train the model on different categories:

```train
bash train.sh
```

To evaluate and test the model, modify the path of models in the main.py and test.sh, and run:

```test
bash test.sh
```

In particular, considering the large differences between the VisA and PCB-Bank dataset and the pre-trained model, we fine-tune VAE of stable diffusion and DINO.
You can refer to the [DiffAD](https://github.com/Loco-Roco/DiffAD) for fine-tuning VAE. 

To fine-tune DINO (referring to [DDAD](https://github.com/arimousa/DDAD)), run:

```fine-tune DINO
python train_dino.py --dataset VisA
```
Quantitative results on MVTec-AD, MPDD, VisA and PCB-Bank datasets. Metrics are I-AUROC/I-AP/I-F1-max at first raw (for detection) and P-AUROC/PAP/P-F1-max/PRO at second raw (for localization).
![image](https://github.com/hyao1/GLAD/assets/52654892/522bf587-8471-4e7e-be3e-c0b080915691)

## Training and Evaluation of the Model for Multi-class
We also test our method at multi-class setting. Pretrained stable diffusion model also is required,  and models (VAE, Unet, DINO) trained by us can be download from [OneDrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/23b903042_stu_hit_edu_cn/Etg1bdDSnOZBt7AydlkCzMUBYKxgmM_9tB-g5M70PJhAVQ).

To train the UNet of stable diffusion, modify the settings in the train.sh and train the model on different categories:

```train
bash train_multi.sh
```

To evaluate and test the model, modify the path of models in the main.py and test.sh, and run:

```test
bash test_multi.sh
```
In particular, we fine-tune VAE of stable diffusion for VisA and PCB-Bank referring to [DiAD](https://github.com/lewandofskee/DiAD).

To fine-tune DINO (referring to [DDAD](https://github.com/arimousa/DDAD)), run:

```fine-tune DINO
python train_dino_multi.py --dataset VisA
```
Quantitative results of multi-category setting on MVTec-AD, MPDD, VisA and PCB-Bank datasets. Metrics are I-AUROC/I-AP/I-F1-max at first raw (for detection) and P-AUROC/P-AP/P-F1-max/PRO at second raw (for localization).
![image](https://github.com/hyao1/GLAD/assets/52654892/b38fe5af-1cb4-4d89-95f3-9ff59f98d96f)

## Training and Evaluation of the Model for Multi-class

The PCB-Bank dataset of the printing circuit board we integrated can be downloaded from here: [PCB-Bank](https://github.com/SSRheart/PCB-Bank)

## Citation

```
@article{yao2024glad,
  title={GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection},
  author={Yao, Hang and Liu, Ming and Wang, Haolin and Yin, Zhicun and Yan, Zifei and Hong, Xiaopeng and Zuo, Wangmeng},
  journal={arXiv preprint arXiv:2406.07487},
  year={2024}
}
```

## Feedback

For any feedback or inquiries, please contact yaohang_1@outlook.com

