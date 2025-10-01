export PYTHONPATH=$(pwd)

# step1: Perform inversion for content video.
CUDA_VISIBLE_DEVICES=1 python src/sd3/run_content_inversion_sd3.py \
                        --content_path examples/contents/mallard-fly \
                        --output_path results/contents-inv \
                        --is_rf_solver # use rf_solver

# step2: Perform inversion for style image.
# CUDA_VISIBLE_DEVICES=1 python src/sd3/run_style_inversion_sd3.py \
#                         --style_path examples/style/00033.png \
#                         --output_path results/style-inv \
#                         --is_rf_solver # use rf_solver

# # step3: Perform mask propagation. [Optional, you can also customize the masks and skip this step.]
# CUDA_VISIBLE_DEVICES=1 python src/mask_propagation.py \
#                         --feature_path results/content-inv/sd3/mallard-fly/features/inversion_feature_map_2_block_301_step.pt \
#                         --backbone 'sd3' \
#                         --mask_path 'examples/mask/mallard-fly.png' \
#                         --output_path 'results/masks'

# # step4: Perform localized video style transfer. [Optional, you can also omit the mask_path to complete the overall style transfer.]
# CUDA_VISIBLE_DEVICES=1 python src/sd3/run_video_style_transfer_sd3.py \
#                         --content_inv_path results/content-inv/sd3/mallard-fly/inversion \
#                         --style_inv_path results/style-inv/sd3/00033/inversion \
#                         --mask_path results/masks/sd3/mallard-fly \
#                         --output_path results/stylization
