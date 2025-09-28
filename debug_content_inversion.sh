export PYTHONPATH=$(pwd)

CUDA_VISIBLE_DEVICES=0 python src/sd3/run_content_inversion_sd3.py \
                        --is_rf_solver \
                        --content_path /data/lxy/sqj/datasets/my_davis2016/data/camel \
                        --output_path /data/lxy/sqj/code/UniVST/sd3_5_tmp
