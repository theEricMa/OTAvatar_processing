python meta_script.py \
    --input_dir ../hdtf_dataset/hdtf_matting/ \
    --output_dir ../hdtf_dataset/hdtf_kp/ \
    --workers 64 \
    --device_ids 0,1,2,3,4,5,6,7 \
    --result-type fg \
    --fps 20