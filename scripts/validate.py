from argparse import ArgumentParser, Namespace

import lightning as L
from omegaconf import OmegaConf

from training_utils.metrics_callback import ModelInspectionCallback
from training_utils.pl_module import PredictionLightningModule


def main(use_gpu: bool, ckpt_path: str) -> None:
    hyperparameters = OmegaConf.load("configs/hyperparameters.yaml")

    module = (
        PredictionLightningModule(
            fast_dev_run=False, hyperparameters=hyperparameters, clearml_task=None
        )
        if not ckpt_path
        else PredictionLightningModule.load_from_checkpoint(
            ckpt_path,
            fast_dev_run=False,
            hyperparameters=hyperparameters,
            clearml_task=None,
            map_location="cpu",
            limit_val_batches=0.1,
        )
    )

    trainer = L.Trainer(
        max_epochs=hyperparameters.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[
            ModelInspectionCallback(
                hyperparameters.viz_scenario_offset, hyperparameters.viz_num_scenarios
            ),
        ],
    )
    trainer.validate(module)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--gpu", action="store_true", help="use gpu")
    parser.add_argument("--ckpt", required=False, type=str, help="Checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.gpu, args.ckpt)
