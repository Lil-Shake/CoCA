CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29500 ../scripts/train_coca.py \
        --config ../config/config_coca.py:hps_v2 \
        --config.run_name "hpsv2_coca" 