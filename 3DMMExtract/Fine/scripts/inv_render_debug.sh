python face_tracker_meta_script.py \
    --input_dir  ../../example_dataset/example_matting \
    --keypoint_dir ../../example_dataset/example_kp \
    --output_dir ../../example_dataset/example_inv_render \
    --debug_dir ../../example_dataset/example_inv_debug \
    --img_h 512 \
    --img_w 512 \
    --workers 56 \
    --device_ids 0,1,2,3,4,5,6 \

