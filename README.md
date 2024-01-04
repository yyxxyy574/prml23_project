# PRML 23 Autumn Final Project
## Train
### vgg16
```
python -m detection.code --train --learning-rate=1e-3 --epochs=15 --load-from=vgg16_caffe.pth --save-best-to=vgg16_15.pth
```
### resnet
```
python -m detection.code --train --learning-rate=1e-3 --epochs=15 --backbone=resnet50 --save-best-to=resnet_50.pth
```
use `--no-augment` for disable data augment
## Predict
```
python -m detection.code --predict --load-from=vgg15_15.pth
```