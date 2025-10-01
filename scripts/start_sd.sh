export PYTHONPATH=$(pwd)

# step1: Perform inversion for content video.
CUDA_VISIBLE_DEVICES=1 python src/sd/run_content_inversion_sd.py \
                        --content_path examples/contents/mallard-fly \
                        --output_path results/contents-inv \
                        --is_opt

# step2: Perform inversion for style image.
CUDA_VISIBLE_DEVICES=1 python src/sd/run_style_inversion_sd.py \
                        --style_path examples/styles/00033.png \
                        --output_path results/styles-inv

# step3: Perform mask propagation. [Optional, you can also customize the masks and skip this step.]
CUDA_VISIBLE_DEVICES=1 python src/mask_propagation.py \
                        --feature_path results/contents-inv/sd/mallard-fly/features/inversion_feature_map_2_block_301_step.pt \
                        --backbone 'sd' \
                        --mask_path 'examples/masks/mallard-fly.png' \
                        --output_path 'results/masks'

# step4: Perform localized video style transfer. [Optional, you can also omit the mask_path to complete the overall style transfer.]
CUDA_VISIBLE_DEVICES=1 python src/sd/run_video_style_transfer_sd.py \
                        --content_inv_path results/contents-inv/sd/mallard-fly/inversion \
                        --style_inv_path results/styles-inv/sd/00033/inversion \
                        --mask_path results/masks/sd/mallard-fly \
                        --output_path results/stylizations
