export PYTHONPATH=$(pwd)

# CUDA_VISIBLE_DEVICES=5 python src/sd/run_content_inversion_sd.py --content_path ./examples/content/rhino \
#                             --output_path ./output-sd-1-5 \
#                             --is_opt

# CUDA_VISIBLE_DEVICES=2 python mask_propagation.py --feature_path ./output/features/libby/inversion_feature_301.pt \
#                             --mask_path ./examples/mask/libby.png \
#                             --output_path ./output

# CUDA_VISIBLE_DEVICES=5 python  src/sd/run_style_inversion_sd.py --style_path ./examples/style/00049.png \
#                         --output_path ./output-sd-1-5

CUDA_VISIBLE_DEVICES=4 python src/sd/run_video_style_transfer_sd.py \
                            --inv_path /data/lxy/sqj/code/UniVST/output-sd-1-5/rhino/inversion \
                            --style_path /data/lxy/sqj/code/UniVST/output-sd-1-5/00049/inversion \
                            --mask_path /data/lxy/sqj/datasets/my_davis2016/mask/rhino \
                            --output_path ./output-sd-1-5
