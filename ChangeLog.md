# YoloV3 PyTorch ChangeLog 

## This is an changelog file for this project
### Updated 20.9.2023 - 16:54

# Version / Changes

## Version 0.1.0

## Version 0.2.0

### Added img_add_noise_blur.py (prog bar)
### Enhancement on train.py (best chkp stats writer)
### Bug fixes on train.py (best chkp stats writer)
### Bug fix on train.py (optimizer usage) 
### Enhancement test.py (higher decimal count on mAP)
### Save best checkpoint evaluation stats


## Version 0.3.0

### ClearML integration
Documentation: https://clear.ml/docs/latest/docs

### Batch size calculation
Documentation: 

### Lr scheduler for warmup
Documentation: 

### GradScaler

### Training warmup
Documentation: https://github.com/Tony-Y/pytorch_warmup

### Multiple fixes on training optimizer
Documentation: https://pytorch.org/docs/master/notes/amp_examples.html

### Save only last checkpoint instead of x number latest

### Implementation of RMSprop optimizer
Documentation: https://github.com/bentrevett/a-tour-of-pytorch-optimizers/blob/main/a-tour-of-pytorch-optimizers.ipynb

# Open issues / bugs:
1. rmsprop optimizer zeroes learning rate too fast
2. *.best.pth is not stored into correct folder (might be runpod issue)
3. Smart optimizer doesn't seem to work corretly -> training takes quite long time