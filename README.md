# OTAvatar_processing
This repository provides tools for preprocessing videos for HDTF dataset used in the [paper](https://github.com/theEricMa/OTAvatar)

# Environment Setup
We are still refining the environment setup at the moment. Please refer to the README file in each subdirectory for guidance. Once we finalize everything, we will gather all the necessary requirements into a requirements.txt file located in the root directory.

# Step 0: Video Download
This section is adopted from [HDTF](https://github.com/MRzzm/HDTF)

# Step 1: Background Removal
This seciton is adopated from [MODNet](https://github.com/ZHKKKe/MODNet)

# Step 2: Video Cropping
This section is adopted from [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing)

# Step 3: Landmark Detection
This section is adopted from [PIRenderer](https://github.com/RenYurui/PIRender)

# 3DMM Extraction
## Step 4: Coarse Stage
This section is adopted from [PIRenderer](https://github.com/RenYurui/PIRender)

## Step 5: Fine Stage
This section is adopted from [ADNeRF](https://github.com/YudongGuo/AD-NeRF)

# Q&A
### Why the Background Removal is performed on the uncropped videos?
During our attempt to remove the background from uncropped videos, we observed that the matting results were unstable between frames. This was due to the fact that the model was trained on images where the head only occupied a small portion of the entire image, which is precisely what is captured in raw video. As a result, we decided to first remove the background and then crop the face, which led to more encouraging results.

### Why the Video Cropping is different from FOMM?
As a talking face processing method, the intitial idea is to crop the 

# Acknowledge
We appreciate [HDTF](https://github.com/MRzzm/HDTF), [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing), [ADNeRF](https://github.com/YudongGuo/AD-NeRF), [PIRenderer](https://github.com/RenYurui/PIRender), [MODNet](https://github.com/ZHKKKe/MODNet) for providing their processing script.
