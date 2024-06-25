# GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection.
The official code of "GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection".
Official implementation of [GLAD](https://arxiv.org/abs/2406.07487)



## Requirements
This repository is implemented and tested on Python 3.10 and PyTorch 2.0.1.
To install requirements:

```setup
pip install -r requirements.txt
```

## Train and Evaluation of the Model
First, you should download the pretrained stable diffusion model from [pretrained model](https://huggingface.co/CompVis/stable-diffusion-v1-4), and datasets. (In addition, [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset is required for anomaly synthesis) 

To train the UNet of stable diffusion, modify the settings in the train.sh and train the model on different categories:

```train
bash train.sh
```
Model checkpoints trained by us are [Baidu Netdisk](https://pan.baidu.com/s/1xH24M4OxAmutRxVPlc011w?pwd=lpk3)

To evaluate and test the model, modify the path of models in the main.py and test.sh, and run:

```test
bash test.sh
```

In particular, considering the large differences between the visa dataset and the pre-trained model, we fine-tune VAE of stable diffusion and DINO.
You can refer to the [DiffAD](https://github.com/Loco-Roco/DiffAD) for fine-tuning VAE. 

To fine-tune DINO, run:

```fine-tune DINO
python train-dino.py --dataset VisA
```
To test DINO, run:

```fine-tune DINO
python test-dino.py --dataset VisA
```
## Code and models about multi-class
coming soon

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

