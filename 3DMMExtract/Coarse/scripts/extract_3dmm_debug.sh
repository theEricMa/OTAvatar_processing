export CUDA_VISIBLE_DEVICES=0
python face_recon_videos.py \
    --input_dir ../../example_dataset/example_matting \
    --keypoint_dir ../../example_dataset/example_kp \
    --output_dir ../../example_dataset/example_3dmm \
    --inference_batch_size 100 \
    --name=model_name \
    --epoch=20 \
    --model facerecon