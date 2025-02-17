from argparse import ArgumentParser, Namespace

import lightning as L
from omegaconf import OmegaConf

from metrics_callback import OnTrainCallback
from prediction_module import LightningModule


def main(use_gpu: bool, ckpt_path: str):

    hyperparameters = OmegaConf.load("configs/hyperparameters.yaml")

    module = (
        LightningModule(False, hyperparameters, clearml_task=None, cosine_t_max=100)
        if not ckpt_path
        else LightningModule.load_from_checkpoint(
            ckpt_path,
            fast_dev_run=False,
            hyperparameters=hyperparameters,
            cosine_t_max=100,
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
            OnTrainCallback(hyperparameters.val_dataset),
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
