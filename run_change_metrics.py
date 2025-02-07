import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs/results/change_metrics", config_name="default.yaml")
def main(config: DictConfig):

    from src.results.change_metrics import compute_change_detections
    
    return compute_change_detections(config)


if __name__ == "__main__":
    main()
