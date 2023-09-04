# OTAvatar_processing
This repository provides tools for preprocessing videos for HDTF dataset used in the [paper](https://github.com/theEricMa/OTAvatar)

# Environment Setup
We are still refining the environment setup at the moment. Please refer to the README file in each subdirectory for guidance. Once we finalize everything, we will gather all the necessary requirements into a requirements.txt file located in the root directory. You can create a conda enviroment first via
```
conda create --name otavatar_processing python=3.9
```
# Step 0: Video Download
This section is adopted from [HDTF](https://github.com/MRzzm/HDTF)

# Step 1: Background Removal
This section is in the `BgRemove` folder, derived from [MODNet](https://github.com/ZHKKKe/MODNet). Its purpose is to remove the background from the source video.

# Step 2: Video Cropping
This section is in the `FaceCrop` folder, adopted from [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing). The primary objective is to zoom in on the facial regions and discard any irrelevant background.

# Step 3: Landmark Detection
This section is in the `LmkDet`, adopted from [Deep3DFaceRecon](https://github.com/sicxu/Deep3DFaceRecon_pytorch). It is for detecting facial landmarks in cropped videos, which is a crucial step for extracting 3DMM coefficients in the subsequent section.

# Step 4: 3DMM Extraction
## Step 4-1: Coarse Stage
This section, located in `3DMMExtract/Coarse`, is based on [PIRenderer](https://github.com/RenYurui/PIRender). It's designed to extract the 3D Morphable Model Coefficients, such as expression and pose, from monocular videos. This module is in line with other models that use 3DMM for talking face generation.

## Step 4-2: Fine Stage
This section is in `3DMMExtract/Fine`. The insight of this step is that learning-based methods, like the previously mentioned Coarse Stage, struggle to deliver stable camera poses required for rendering. However, the [ADNeRF](https://github.com/YudongGuo/AD-NeRF) approach suggests utilizing an optimization-based method, which significantly enhances the stability of the camera pose and improves the animation. The trade-off is that this method can be time-intensive. For example, processing a video comprised of numerous sections may take up to an hour.

# Step 5: Dataset compression
This segment in `LmdbPrepare`, is derived from [PIRenderer](https://github.com/RenYurui/PIRender). It employs an Lmdb compressed file format to significantly reduce I/O time and accelerate the training process.

# Q&A
### Why the Background Removal is performed on the uncropped videos?
During our attempt to remove the background from uncropped videos, we observed that the matting results were unstable between frames. This was due to the fact that the model was trained on images where the head only occupied a small portion of the entire image, which is precisely what is captured in raw video. As a result, we decided to first remove the background and then crop the face, which led to more encouraging results.

### Why Video Cropping is different from FOMM?
Initially, we planned to implement [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing), which has been widely utilized in numerous prior methods, as a means of talking face processing. However, we encountered difficulties in obtaining a stable and accurate head pose during the fine stage, as the head occupied too much space in the image. As a result, we opted to make some minor adjustments to our approach.

# Acknowledge
We appreciate [HDTF](https://github.com/MRzzm/HDTF), [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing), [ADNeRF](https://github.com/YudongGuo/AD-NeRF), [PIRenderer](https://github.com/RenYurui/PIRender), [MODNet](https://github.com/ZHKKKe/MODNet) for providing their processing script.
