import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def compressibility():
    config = base.get_config()

    config.pretrained.model = "CompVis/stable-diffusion-v1-4"

    config.num_epochs = 100
    config.use_lora = True
    config.save_freq = 1
    config.num_checkpoint_limit = 100000000

    # the DGX machine I used had 8 GPUs, so this corresponds to 8 * 8 * 4 = 256 samples per epoch. 4 * 16 * 4 = 256
    config.sample.batch_size = 4 # 8 for 8*A100, try 4 for 4*4090
    config.sample.num_batches_per_epoch = 16 # 4 for 8*A100, 16 for 4*4090

    # this corresponds to (8 * 4) / (4 * 2) = 4 gradient updates per epoch. (4 * 16) / (1 * 16) = 4
    config.train.batch_size = 1 # 4 for 8*A100, try 1 for 4*4090
    config.train.gradient_accumulation_steps = 16  # 2 for 8*A100，try 16 for 4*4090

    # prompting
    config.prompt_fn = "imagenet_animals"
    config.prompt_fn_kwargs = {}

    # rewards
    config.reward_fn = "jpeg_compressibility"

    config.per_prompt_stat_tracking = {
        "buffer_size": 16,
        "min_count": 16,
    }

    return config

def aesthetic():
    config = compressibility()
    config.num_epochs = 100
    config.train.clip_range = 5e-5
    
    config.train.gradient_accumulation_steps = 32  # 2 for 8*A100，try 16 for 4*4090
    config.reward_fn = "aesthetic_score"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    # config.train.gradient_accumulation_steps = 32 # 4 for 8*A100, 32 for 4090

    config.prompt_fn = "simple_animals"
    config.per_prompt_stat_tracking = {
        "buffer_size": 32, 
        "min_count": 16,
    }
    return config

def hps_v2():
    config = compressibility()
    config.run_name = "hps_v2_finetuned"
    config.prompt_fn = "hps_v2_all"
    config.reward_fn = "hps_v2"

    config.num_epochs = 100
    # for this experiment, I reserved 2 GPUs for LLaVA inference so only 6 could be used for DDPO. the total number of
    # samples per epoch is 8 * 6 * 6 = 288.
    config.sample.batch_size = 2 # 8
    config.sample.num_batches_per_epoch = 32 # 6

    config.per_prompt_stat_tracking = {
        "buffer_size": 32, 
        "min_count": 16,
    }
    return config

def pickscore():
    config = compressibility()
    config.run_name = "PickScore_finetuned"
    config.reward_fn = "PickScore"
    config.prompt_fn = "simple_animals"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    # config.train.gradient_accumulation_steps = 32 # 4 for 8*A100, 32 for 4090    
    config.per_prompt_stat_tracking = {
        "buffer_size": 32, 
        "min_count": 16,
    }
    return config

def irscore():
    config = compressibility()
    config.run_name = "ImageReward_finetuned"
    config.reward_fn = "ImageReward"
    config.prompt_fn = "simple_animals"

    # this reward is a bit harder to optimize, so I used 2 gradient updates per epoch.
    # config.train.gradient_accumulation_steps = 32 # 4 for 8*A100, 32 for 4090    
    config.per_prompt_stat_tracking = {
        "buffer_size": 32, 
        "min_count": 16,
    }
    return config

def get_config(name):
    return globals()[name]()
