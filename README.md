# GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection.
The official code of "GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection".
Paper [GLAD](https://arxiv.org/abs/2406.07487)

![image](https://github.com/hyao1/GLAD/assets/52654892/62a8d52d-72ab-4bda-8fb4-41b5d8e0a044)


## Requirements
This repository is implemented and tested on Python 3.10 and PyTorch 2.0.1.
To install requirements:

```setup
pip install -r requirements.txt
```

## Models trained by us
Models (VAE, Unet, DINO) trained by us are here: [OneDrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/23b903042_stu_hit_edu_cn/Etg1bdDSnOZBt7AydlkCzMUBYKxgmM_9tB-g5M70PJhAVQ).

## Training and Evaluation of the Model
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

## Code and models about multi-class
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

