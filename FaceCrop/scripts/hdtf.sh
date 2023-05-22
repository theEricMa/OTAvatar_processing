python crop_hdtf.py \
    --workers 7 \
    --device_ids 0,1,2,3,4,5,6 \
    --format .mp4 \
    --chunks_metadata hdtf-metadata-1.csv \
    --image_shape 512,512 \
    --in_folder hdtf_matting_raw \
    --out_folder hdtf_matting \
    --increase 0.1

python crop_hdtf.py \
    --workers 7 \
    --device_ids 0,1,2,3,4,5,6 \
    --format .mp4 \
    --chunks_metadata hdtf-metadata-2.csv \
    --image_shape 512,512 \
    --in_folder hdtf_matting_raw \
    --out_folder hdtf_matting \
    --increase 0.2