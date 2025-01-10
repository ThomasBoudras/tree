import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs/results/change_metrics", config_name="default.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.results.change_metrics import compute_change_metrics

    # Train model
    return compute_change_metrics(config)


if __name__ == "__main__":
    main()
