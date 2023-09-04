python prepare_lmdb.py \
    --path  ../hdtf_dataset/hdtf_matting \
    --keypoint_path ../hdtf_dataset/hdtf_kp/ \
    --coeff_3dmm_path ../hdtf_dataset/hdtf_3dmm/ \
    --inv_3dmm_path ../hdtf_dataset/hdtf_inv_render \
    --out ../hdtf_dataset/hdtf_lmdb  \
    --n_worker 20 \
    --sizes 512 