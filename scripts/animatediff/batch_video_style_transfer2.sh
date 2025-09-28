export PYTHONPATH=$(pwd)


# for tgve-laion
# inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/animatediff-easyinv/tgve
# style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/animatediff/laion
# mask_path=/data/lxy/sqj/datasets/my_tgve/mask
# output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/animatediff

# CUDA_VISIBLE_DEVICES=2 python src/animatediff/run_video_style_transfer_animatediff.py \
#                             --inv_path $inv_path \
#                             --style_path $style_path \
#                             --mask_path $mask_path \
#                             --output_path $output_path

# for tgve-wikiart
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/animatediff-easyinv/tgve
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/animatediff/wikiart
mask_path=/data/lxy/sqj/datasets/my_tgve/mask/
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/animatediff

CUDA_VISIBLE_DEVICES=2 python src/animatediff/run_video_style_transfer_animatediff.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path