# OTAvatar_processing
This repository provides tools for preprocessing videos for HDTF dataset used in the [paper](https://github.com/theEricMa/OTAvatar)

This repository offers a set of tools designed to pre-process videos for the HDTF dataset, which is featured in the associated research paper.

# Installation
## Environment setup
1. Set up a conda environment with all dependencies as follows:
```
git clone https://github.com/theEricMa/OTAvatar_processing.git
cd OTAvatar_processing
conda create --name otavatar_processing python=3.9
conda activate otavatar_processing
```
2. Install pytorch3d library
```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e . && cd ..
```
## Prepare prerequisite models

### [MODNet](https://github.com/theEricMa/OTAvatar_processing/blob/main/BgRemove)
   
See [BgRemove](https://github.com/theEricMa/OTAvatar_processing/tree/main/BgRemove) folder for details. 

### [Bace Face Model](https://github.com/sicxu/Deep3DFaceRecon_pytorch#:~:text=Basel%20Face%20Model%202009%20(BFM09)) and [Deep3dFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch)

Adhere to the guidance provided in the [3DMMExtract/Coarse](https://github.com/theEricMa/OTAvatar_processing/tree/main/3DMMExtract/Coarse) directory to download the necessary files. Additionally, duplicate some of these files into the [3DMMExtract/Fine](https://github.com/theEricMa/OTAvatar_processing/tree/main/3DMMExtract/Fine) folder.

# Step 0: Split Videos
Generally, the video is suggested in the following format: 
```
<your_dataset_name>
├── <video_split>
│  ├── train
│  │  ├── <video_1.mp4>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.mp4>
│  │  ├── ...
```

# Step 1: Remove Background
This section resides in the in the [BgRemove](https://github.com/theEricMa/OTAvatar_processing/tree/main/BgRemove) folder, which is based on [MODNet](https://github.com/ZHKKKe/MODNet). Its primary function is to eliminate the background from the input video. The process will create the subsequent subdirectory:
```
<your_dataset_name>
├── <video_split>
├── <video_matting_raw> <<- new
│  ├── train
│  │  ├── <video_1.mp4>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.mp4>
│  │  ├── ...
```

# Step 2: Crop Videos
The following section is found within the [FaceCrop](https://github.com/theEricMa/OTAvatar_processing/tree/main/FaceCrop) folder, which was adapted from the [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing). Its main purpose is to focus on the facial areas of the video and eliminate any unnecessary background. This procedure will result in the creation of a new subdirectory:
```
<your_dataset_name>
├── <video_split>
├── <video_matting_raw>
├── <video_matting>     <<- new
│  ├── train
│  │  ├── <video_1.mp4>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.mp4>
│  │  ├── ...
```

# Step 3: Detect Landmarks
The following segment in [LmkDet](https://github.com/theEricMa/OTAvatar_processing/tree/main/LmkDet) folder, is derived from [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch). This section is for detecting facial landmarks in trimmed videos, an essential step for obtaining 3DMM coefficients in the following section. This procedure will generate the next subdirectory:
```
<your_dataset_name>
├── <video_split>
├── <video_matting_raw>
├── <video_matting>
├── <video_kp>         <<- new
│  ├── train
│  │  ├── <video_1.txt>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.txt>
│  │  ├── ...
```

# Step 4: Extract 3DMM
## Step 4-1: Coarse Stage
This section, located in [3DMMExtract/Coarse](https://github.com/theEricMa/OTAvatar_processing/tree/main/3DMMExtract/Coarse), is based on [PIRenderer](https://github.com/RenYurui/PIRender). It's designed to extract the 3D Morphable Model Coefficients, such as expression and pose, from monocular videos. This module is in line with other models that use 3DMM for talking face generation. This procedure will result in the creation of a new subdirectory:
```
<your_dataset_name>
├── <video_split>
├── <video_matting_raw>
├── <video_matting>
├── <video_kp>
├── <video_3dmm>       <<- new
│  ├── train
│  │  ├── <video_1.mat>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.mat>
│  │  ├── ...
```

## Step 4-2: Fine Stage
This section is found under [3DMMExtract/Fine](https://github.com/theEricMa/OTAvatar_processing/tree/main/3DMMExtract/Coarse). The insight of this step is that learning-based methods, like the previously mentioned Coarse Stage, struggle to deliver stable camera poses required for rendering. However, the [ADNeRF](https://github.com/YudongGuo/AD-NeRF) approach suggests utilizing an optimization-based method, which significantly enhances the stability of the camera pose and improves the animation. The trade-off is that this method can be time-intensive. For example, processing a video comprised of numerous sections may take up to an hour. This procedure will result in the creation of a new subdirectory. This process will lead to the creation of two new subdirectories:
```
<your_dataset_name>
├── <video_split>
├── <video_matting_raw>
├── <video_matting>
├── <video_kp>
├── <video_3dmm>
├── <video_inv_render>  <<-new
│  ├── train
│  │  ├── <video_1.mat>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.mat>
│  │  ├── ...
├── <video_inv_debug>  <<-new #just for debug
│  ├── train
│  │  ├── <video_1.mp4>
│  │  ├── ...
│  ├── test
│  │  ├── <video_1.mp4>
│  │  ├── ...
```

# Step 5: Compress Dataset
The following portion in [LmdbPrepare](https://github.com/theEricMa/OTAvatar_processing/tree/main/LmdbPrepare) is adapted from [PIRenderer](https://github.com/RenYurui/PIRender). It employs an Lmdb compressed file format to significantly reduce I/O time and accelerate the training process. This procedure will result in the creation of a new subdirectory:
```
<your_dataset_name>
├── <video_split>
├── <video_matting_raw>
├── <video_matting>
├── <video_kp>
├── <video_3dmm>
├── <video_inv_render>
├── <video_inv_debug>
├── <video_lmdb>       <<- new
│  │  ├── <resolution>
│  │  │  ├── data.mdb
│  │  │  ├── lock.mdb
│  │  ├── train_list.txt
│  │  ├── test_list.txt
```

# Q&A
### Why the Background Removal is performed on the uncropped videos?
During our attempt to remove the background from uncropped videos, we observed that the matting results were unstable between frames. This was due to the fact that the model was trained on images where the head only occupied a small portion of the entire image, which is precisely what is captured in raw video. As a result, we decided to first remove the background and then crop the face, which led to more encouraging results.

### Why Video Cropping is different from FOMM?
Initially, we planned to implement [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing), which has been widely utilized in numerous prior methods, as a means of talking face processing. However, we encountered difficulties in obtaining a stable and accurate head pose during the fine stage, as the head occupied too much space in the image. As a result, we opted to make some minor adjustments to our approach.

# Acknowledge
We appreciate [HDTF](https://github.com/MRzzm/HDTF), [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing), [ADNeRF](https://github.com/YudongGuo/AD-NeRF), [PIRenderer](https://github.com/RenYurui/PIRender), [MODNet](https://github.com/ZHKKKe/MODNet) for providing their processing script.
