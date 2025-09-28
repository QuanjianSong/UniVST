export PYTHONPATH=$(pwd)

content_path='/data/lxy/sqj/datasets/my_davis2016/data'
name=davis2016

CUDA_VISIBLE_DEVICES=3 python src/sdxl/run_content_inversion_sdxl.py --content_path=$content_path \
                            --output_path ./outputs/content-inv/sdxl-easyinv/$name \
                            --is_opt


content_path='/data/lxy/sqj/datasets/my_tgve/data'
name=tgve

CUDA_VISIBLE_DEVICES=3 python src/sdxl/run_content_inversion_sdxl.py --content_path=$content_path \
                            --output_path ./outputs/content-inv/sdxl-easyinv/$name \
                            --is_opt