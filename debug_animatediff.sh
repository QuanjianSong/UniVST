export PYTHONPATH=$(pwd)

# CUDA_VISIBLE_DEVICES=4 python src/animatediff/run_content_inversion_animatediff.py --content_path ./examples/content/rhino \
#                             --output_path ./output-animatediff-dreambooth \
#                             --is_opt

# CUDA_VISIBLE_DEVICES=2 python mask_propagation.py --feature_path ./output/features/libby/inversion_feature_301.pt \
#                             --mask_path ./examples/mask/libby.png \
#                             --output_dir ./output

# CUDA_VISIBLE_DEVICES=4 python src/animatediff/run_style_inversion_animatediff.py --style_path examples/style/style1.png \
#                         --output_path ./output-animatediff-dreambooth \

CUDA_VISIBLE_DEVICES=4 python src/animatediff/run_video_style_transfer_animatediff.py \
                            --inv_path ./outputs/content-inv/animatediff/davis2016/camel/inversion \
                            --style_path ./outputs/style-inv/animatediff/wikiart/00000/inversion \
                            --mask_path /data/lxy/sqj/datasets/my_davis2016/mask/camel \
                            --output_path ./output-animatediff
