# Land-segmentation
## Goal
This repository is intended to make an attempt on doing semantic land segmentation, by generating image masks

## Background
The training dataset used is the DeepGlobe landsegmentation dataset

### Installation
In order to run you have to have python installed 3.10 is the recommended.
It is also recommended to create a virtual environment.
If the above is complete follow the steps below:
```
pip install -r requirements.txt
```
this will install all the required dependencies
```
python shape.py
```
This will load in and resize the default archive and resize it
```
python train.py
```
This will run the training on the choosen model, and save it after training
```
python test.py
```
This will load in the validation images in matplotlib to compare results

## Issues
If you notice an issue place submit a ticket, or if you think you solved the issue please create a pull request.
