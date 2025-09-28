style_path='/data/lxy/sqj/datasets/my_laion'
name=laion

CUDA_VISIBLE_DEVICES=4 python run_style_inversion_sd.py --style_path $style_path \
                            --output_path ./outputs/style-inv/sd2-1/$name \


style_path='/data/lxy/sqj/datasets/my_wikiart'
name=wikiart

CUDA_VISIBLE_DEVICES=4 python run_style_inversion_sd.py --style_path $style_path \
                            --output_path ./outputs/style-inv/sd2-1/$name \
