export PYTHONPATH=$(pwd)

style_path='/data/lxy/sqj/datasets/my_laion'
name=laion

CUDA_VISIBLE_DEVICES=4 python src/sdxl/run_style_inversion_sdxl.py --style_path $style_path \
                            --output_path ./outputs/style-inv/sdxl/$name \


style_path='/data/lxy/sqj/datasets/my_wikiart'
name=wikiart

CUDA_VISIBLE_DEVICES=4 python src/sdxl/run_style_inversion_sdxl.py --style_path $style_path \
                            --output_path ./outputs/style-inv/sdxl/$name \
