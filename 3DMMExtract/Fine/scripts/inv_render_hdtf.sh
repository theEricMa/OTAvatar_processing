python face_tracker_meta_script.py \
    --input_dir  ../../hdtf_dataset/hdtf_matting \
    --keypoint_dir ../../hdtf_dataset/hdtf_kp \
    --output_dir ../../hdtf_dataset/hdtf_inv_render \
    --debug_dir ../../hdtf_dataset/hdtf_inv_debug \
    --img_h 512 \
    --img_w 512 \
    --workers 56 \
    --device_ids 0,1,2,3,4,5,6 \



