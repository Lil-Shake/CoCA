CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29500 ../scripts/train_uca.py \
        --config ../config/config_coca.py:aesthetic \
        --config.run_name "aesthetic_100_uca" 