python prepare_lmdb.py \
    --path  ../example_dataset/example_matting \
    --keypoint_path ../example_dataset/example_kp/ \
    --coeff_3dmm_path ../example_dataset/example_3dmm/ \
    --inv_3dmm_path ../example_dataset/example_inv_render \
    --out ../example_dataset/example_lmdb  \
    --n_worker 20 \
    --sizes 512 