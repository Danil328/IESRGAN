# IESRGAN
Improved ESRGAN

1. Fill the config
```
config.json
```

2. Prepare data
```
dataset/prepare_data.sh
```
3. Train Generator with features loss
```
cd src
python train_PSNR_model.py
```
4. GAN
```
cd src
python train.py
```
