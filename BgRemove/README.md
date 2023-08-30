# BgRemove

## Environment Setup
```
pip install -r requirements.txt
```
## Model Download
Create the folder `pretrained` under the current directory. Download the model weights `modnet_webcam_portrait_matting.ckpt` from  this [link](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing) and save it in the `pretrained` folder.

## Environment Debug
Run the following script to check if your environment is set up perfectly.
```
sh scripts/matting_debug.sh
```

## Start Processing
Execute the script below and modify the arguments, such as `device_ids` and `workers`, according to your devices.
```
sh scripts/matting_hdtf.sh
```
