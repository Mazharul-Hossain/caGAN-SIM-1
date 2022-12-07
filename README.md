# caGAN-SIM

- forked from [qc17-THU/caGAN-SIM](https://github.com/qc17-THU/caGAN-SIM)

**caGAN-SIM modified software** is a tensorflow implementation for deep learning-based 3D-SIM reconstruction. This repository is modiefied from a development based on the 2021 IEEE JSTQE paper [**3D Structured Illumination Microscopy via Channel Attention Generative Adversarial Network**](https://doi.org/10.1109/JSTQE.2021.3060762).<br>

University of Memphis, TN, USA.<br>

## Contents

- [caGAN-SIM](#cagan-sim)
  - [Contents](#contents)
  - [Advanced Topics in Machine Learning](#advanced-topics-in-machine-learning)
  - [Environment](#environment)
  - [File structure](#file-structure)
  - [Test pre-trained models](#test-pre-trained-models)
  - [Train a new model](#train-a-new-model)
  - [License](#license)
  - [Citation](#citation)

## Advanced Topics in Machine Learning  

This is the Final Project of Advanced Topics in Machine Learning (COMP-7747, COMP-8747) Fall Term 2022.

## Environment

- Ubuntu 20.04
- CUDA 11.0.207
- cudnn 8.0.4
- Python 3.6.10
- Tensorflow 2.4.0
- GPU: GeForce RTX 2080Ti

We have trained and tested on similar environment.  
Our training environment is as below:  

```cmd
  # Create a new environment
  #     $ conda create --name caGAN_SIM python=3.7
  #
  # To activate this environment, use
  #     $ conda activate caGAN_SIM
  #
  # To deactivate an active environment, use
  #     $ conda deactivate
  #
  # To remove this environment, use
  #     $ conda env remove -n caGAN_SIM
  # 
  # Install required packages 
  # 
  #     $ conda install -y tensorflow-gpu=2.4
        $ conda install -y -c conda-forge matplotlib imageio tifffile
        $ conda install -y -c anaconda scikit-image
        $ python -m pip install --upgrade opencv-contrib-python sewar
```

## File structure

- `./dataset` is the default path for training data and testing data
  - `./dataset/training` The augmented training image patch pairs will be saved here by default
  - `./dataset/training_gt` The high resolution ground truth training image will be saved here by default
  - `./dataset/validate` The augmented validation image patch pairs will be saved here by default
  - `./dataset/validate_gt` The high resolution ground truth validation image patch pairs will be saved here by default
  - `./dataset/test` includes some demo images of F-actin and microtubules to test caGAN-SIM models
- `./src` includes the source codes of caGAN-SIM
  - `./src/models` includes declaration of caGAN models
  - `./src/utils` is the tool package of caGAN-SIM software
- `./trained_models` place pre-trained caGAN-SIM models here for testing, and newly trained models will be saved here by default

## Test pre-trained models

- Place your testing data in `./dataset/test`
- Download [pre-trained model](https://gofile.io/d/hlMyWd) and place it under `./trained_models/[save_weights_name]`
- Open your terminal and cd to `./src`
- Run `bash demo_predict.sh` in your terminal. Note that before running the bash file, you should check if the data paths and other arguments in `demo_predict.sh` are set correctly
- The output reconstructed SR images will be saved in `--data_dir`

## Train a new model

- Data for training: You can train a new caGAN-SIM model using your own datasets. Note that you'd better divide the dataset of each specimen into training part and validation/testing part before training, so that you can test your model with the preserved validation/testing data
- Run `bash demo_train.sh` in your terminal to train a new caGAN-SIM model. Similar to testing, before running the bash file, you should check if the data paths and the arguments are set correctly
- You can run `tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph` to monitor the training process via tensorboard. If the validation loss isn't likely to decay any more, you can use early stop strategy to end the training
- Model weights will be saved in `./trained_models/` by default

## License

This repository is released under the MIT License (refer to the LICENSE file for details).

