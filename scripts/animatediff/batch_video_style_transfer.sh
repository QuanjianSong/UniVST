export PYTHONPATH=$(pwd)


# for davis-laion
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/animatediff-easyinv/davis2016
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/animatediff/laion
mask_path=/data/lxy/sqj/datasets/my_davis2016/mask
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/animatediff

CUDA_VISIBLE_DEVICES=4 python src/animatediff/run_video_style_transfer_animatediff.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path

# for davis-wikiart
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/animatediff-easyinv/davis2016
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/animatediff/wikiart
mask_path=/data/lxy/sqj/datasets/my_davis2016/mask/
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/animatediff

CUDA_VISIBLE_DEVICES=4 python src/animatediff/run_video_style_transfer_animatediff.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path