export PYTHONPATH=$(pwd)


# for tgve-laion
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/sdxl-easyinv/tgve
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/sdxl/laion
mask_path=/data/lxy/sqj/datasets/my_tgve/mask
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/sdxl

CUDA_VISIBLE_DEVICES=5 python src/sdxl/run_video_style_transfer_sdxl.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path


# for tgve-wikiart
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/sdxl-easyinv/tgve
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/sdxl/wikiart
mask_path=/data/lxy/sqj/datasets/my_tgve/mask/
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/sdxl

CUDA_VISIBLE_DEVICES=5 python src/sdxl/run_video_style_transfer_sdxl.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path