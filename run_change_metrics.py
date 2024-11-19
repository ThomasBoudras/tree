import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs/experiment", config_name="compute_change_detection_metrics_config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.changes_detection.compute_change_detection_metrics import compute_change_detection

    # Train model
    return compute_change_detection(config)


if __name__ == "__main__":
    main()
