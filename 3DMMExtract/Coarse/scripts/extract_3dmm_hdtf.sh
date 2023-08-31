export CUDA_VISIBLE_DEVICES=0
python face_recon_videos.py \
    --input_dir ../../hdtf_dataset/hdtf_matting \
    --keypoint_dir ../../hdtf_dataset/hdtf_kp \
    --output_dir ../../hdtf_dataset/hdtf_3dmm \
    --inference_batch_size 100 \
    --name=model_name \
    --epoch=20 \
    --model facerecon