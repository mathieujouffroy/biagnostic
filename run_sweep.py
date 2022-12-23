import wandb
import yaml
#from absl import app
from functools import partial
from run_training import run_model
from train_framework.utils import parse_args, YamlNamespace
#FLAGS = flags.FLAGS


def main():
    config, sweep_config = parse_args()

    sweep_id = wandb.sweep(
        sweep_config,
        project=config.project_name,
    )

    #wandb.agent(sweep_id, function=partial(run_model, config), count=10)
    wandb.agent(sweep_id, function=partial(run_model, config), count=8)


if __name__ == "__main__":
    #app.run(main)
    main()