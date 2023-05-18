# OTAvatar_processing
This repository provides tools for preprocessing videos for HDTF dataset used in the [paper](https://github.com/theEricMa/OTAvatar)

# Environment Setup
```
conda create --name otavatar_processing python=3.9
```

# Video Download
This section is adopted from [HDTF](https://github.com/MRzzm/HDTF)

# Background Removal
This seciton is adopated from [MODNet](https://github.com/ZHKKKe/MODNet)

# Video Cropping
This section is adopted from [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing)

# Landmark Detection
This section is adopted from [PIRenderer](https://github.com/RenYurui/PIRender)

# 3DMM Extraction
## Coarse Stage
This section is adopted from [PIRenderer](https://github.com/RenYurui/PIRender)

## Fine Stage
This section is adopted from [ADNeRF](https://github.com/YudongGuo/AD-NeRF)

# Q&A
### Why the Background Removal is performed on the uncropped videos?
When we tried to remove the background from uncropped videos, we noticed that the results were often unstable and not very promising. The reason for this is that the model was trained on images where the head only occupies a small portion of the entire image, which is exactly what is captured in raw video. Therefore, we began by removing the background and then cropping the face, which resulted in more promising outcomes.

# Acknowledge
We appreciate [HDTF](https://github.com/MRzzm/HDTF), [FOMM](https://github.com/AliaksandrSiarohin/video-preprocessing), [ADNeRF](https://github.com/YudongGuo/AD-NeRF), [PIRenderer](https://github.com/RenYurui/PIRender), [MODNet](https://github.com/ZHKKKe/MODNet) for providing their processing script.
