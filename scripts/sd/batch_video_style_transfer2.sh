
# for tgve-laion
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/sd2-1-easyinv/tgve
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/sd2-1/laion
mask_path=/data/lxy/sqj/datasets/my_tgve/mask
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/sd2-1

CUDA_VISIBLE_DEVICES=4 python video_style_transfer_sd.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path

# for tgve-wikiart
inv_path=/data/lxy/sqj/code/UniVST/outputs/content-inv/sd2-1-easyinv/tgve
style_path=/data/lxy/sqj/code/UniVST/outputs/style-inv/sd2-1/wikiart
mask_path=/data/lxy/sqj/datasets/my_tgve/mask/
output_path=/data/lxy/sqj/code/UniVST/outputs/stylized/sd2-1

CUDA_VISIBLE_DEVICES=4 python video_style_transfer_sd.py \
                            --inv_path $inv_path \
                            --style_path $style_path \
                            --mask_path $mask_path \
                            --output_path $output_path