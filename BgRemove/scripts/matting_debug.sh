python meta_script.py \
    --input_dir ../example_dataset/example_split \
    --output_dir ../example_dataset/example_matting_raw \
    --workers 20 \
    --device_ids 0,1,2,3,4,5 \
    --result-type fg \
    --fps 20
