# Land-segmentation
## Goal
This repository is intended to make an attempt on doing semantic land segmentation, by generating image masks. The goal is to make an neural network that can do on board satellite image processing and feature recognition, such that the long term goal is to shrink the models size to as small as possible.

## Background
The training dataset used is the DeepGlobe landsegmentation dataset, this is further shaped and augmented during training. The library is based on python 3.10 and tensorflow 2.10

### Installation
In order to run the shaping, training, and test you have to have python installed, 3.10 is the recommended.
It is also recommended to create a virtual environment, with the above python version.
If the above steps are complete follow the steps below:
```
pip install -r requirements.txt
```
this will install all the required dependencies
```
python shape.py
```
This will load in and resize the default archive and resize it.
```
python train.py
```
This will run the training on the choosen model, and save it after training.
Keep in mind if the model is set to unknown, as of know, it has to be trained through
```
python test_generator.py
```
This is because the prior uses, a library keras-segmentation, the later is our own implementation, such two separate set of training tools are required.

```
python test.py
```
This will load in the validation images in matplotlib to compare results

## Issues
If you notice an issue place submit a ticket, or if you think you solved the issue please create a pull request.
