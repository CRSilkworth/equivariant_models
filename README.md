![paper_image](https://github.com/CRSilkworth/equivariant_models/blob/master/paper_images.png)
# Equivariant models
This repository holds the code used in the evaluation section of [this paper](https://github.com/CRSilkworth/equivariant_models/blob/master/invariance_equivariance_weight_sharing.pdf). The [image_net](https://github.com/CRSilkworth/equivariant_models/tree/master/image_net) directory holds all the code use in the preprocessing and conversion of the images into tfrecords, while the [alex_net](https://github.com/CRSilkworth/equivariant_models/tree/master/alex_net) holds the various model definitions of the original alex net and the flip invariant version. It also holds the training script that was used to get the evaluation results in the paper. The [constrained_weights](https://github.com/CRSilkworth/equivariant_models/tree/master/constrained_weights) holds the code that actually enacts a flip invariant layer. 

## Install and running
The image net data is required in order to run this pipeline. 
Install:
```
pip install https://github.com/CRSilkworth/equivariant_models.git
```
or 
```
git clone https://github.com/CRSilkworth/equivariant_models
cd equivariant_models
export PYTHONPATH=$PWD
```
Alter equivariant_models/image_net/cfgs/write_images_to_tfrecords_cfg.py to point to the appropriate directories. Note this script can take a while.
```
cd image_net
python write_images_to_tfrecords.py cfgs/write_images_to_tfrecords_cfg.py
```
Alter equivariant_models/alex_net/cfgs/train_cfg.py to point to the appropriate directories. The variable 'flip_constrain_fc6' is what controlls whether or not a flip invariant layer is used. Note that training can take several days, depending on your GPU
```
cd equivariant_models/alex_net/
python train.py cfgs/train_cfg.py
```
Tensorboard can be used to monitor the training progress.
