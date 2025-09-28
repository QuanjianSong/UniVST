content_path='/data/lxy/sqj/datasets/my_davis2016/data'
name=davis2016

CUDA_VISIBLE_DEVICES=2 python run_content_inversion_sd.py --content_path=$content_path \
                            --output_path ./outputs/content-inv/sd2-1/$name


content_path='/data/lxy/sqj/datasets/my_tgve/data'
name=tgve

CUDA_VISIBLE_DEVICES=2 python run_content_inversion_sd.py --content_path=$content_path \
                            --output_path ./outputs/content-inv/sd2-1/$name