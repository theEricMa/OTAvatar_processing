python meta_script.py \
    --input_dir ../hdtf_dataset/HDTF_split/ \
    --output_dir ../hdtf_dataset/HDTF_matting_raw/ \
    --workers 64 \
    --device_ids 0,1,2,3,4,5,6,7 \
    --result-type fg \
    --fps 20