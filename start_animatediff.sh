export PYTHONPATH=$(pwd)

# step1: Perform inversion for content video.
CUDA_VISIBLE_DEVICES=1 python src/animatediff/run_content_inversion_animatediff.py \
                        --content_path examples/content/mallard-fly \
                        --output_path results/content-inv \
                        --is_opt

# step2: Perform inversion for style image.
CUDA_VISIBLE_DEVICES=1 python src/animatediff/run_style_inversion_animatediff.py \
                        --style_path examples/style/00033.png \
                        --output_path results/style-inv \

# step3: Perform mask propagation. [Optional, you can also customize the masks and skip this step.]
CUDA_VISIBLE_DEVICES=1 python src/mask_propagation.py \
                        --feature_path results/content-inv/animatediff/mallard-fly/features/inversion_feature_map_2_block_301_step.pt \
                        --backbone 'animatediff' \
                        --mask_path 'examples/mask/mallard-fly.png' \
                        --output_path 'results/masks'

# step4: Perform localized video style transfer. [Optional, you can also omit the mask_path to complete the overall style transfer.]
CUDA_VISIBLE_DEVICES=1 python src/animatediff/run_video_style_transfer_animatediff.py \
                        --content_inv_path results/content-inv/animatediff/mallard-fly/inversion \
                        --style_inv_path results/style-inv/animatediff/00033/inversion \
                        --mask_path results/masks/animatediff/mallard-fly \
                        --output_path results/stylization
