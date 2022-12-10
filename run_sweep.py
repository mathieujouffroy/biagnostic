import wandb
import yaml
#from absl import app
from functools import partial
from run_training import run_model
from train_framework.utils import parse_args
#FLAGS = flags.FLAGS

#config, sweep_config = parse_args()
#print("BAM")

def main():
    print("bim")
    config, sweep_config = parse_args()
    print(config)
    print(sweep_config)
    print("bim")

    #sweep_config = {
    #    "method": "bayes", # random
    #    "metric": {"name": "val_loss", "goal": "minimize"},
    #    "early_terminate": {
    #        "type": "hyperband",
    #        "min_iter": 5,
    #    },
    #    "parameters": {
    #        "n_epochs": {"values": 1}, # 10
#
    #        "batch_size": {"values": [4, 5, 6]},
    #        
    #        "optimizer": {"values": ["Adam", "AdamW", "SGD"]},
    #        
    #        "learning_rate": {"distribution": "uniform", "min": 1e-5, "max": 5e-3},
    #        "weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.02},
    #        
    #        "lr_scheduler":{
    #            "values": [None,  "cosine", "polynomial", "exponential", 
    #                    "time", "plateau_reduce","tf_cosine", "tf_exp", "tf_invtime"]
    #        },
    #        "lr_decay": {"distribution": "uniform", "min": 0.0, "max": 0.5},
    #        
    #    },
    #}
    sweep_id = wandb.sweep(
        sweep_config,
        project=config.project_name,
    )
    wandb.agent(sweep_id, function=partial(run_model, config), count=10)
    #wandb.agent(sweep_id, function=(run_model, config), count=10)


if __name__ == "__main__":
    #app.run(main)
    main()