import warnings

# Ignore the warning about nested tensors to not spam the terminal
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.",
)

import typing as T
from argparse import ArgumentParser, Namespace

import lightning as L
import matplotlib.pyplot as plt
import torch
from clearml import Task
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner import Tuner
from omegaconf import OmegaConf
from pytorch_lightning.profilers import SimpleProfiler

from metrics_callback import OnTrainCallback
from prediction_module import LightningModule

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("medium")

LR_FIND = False


task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction MCG")


def main(fast_dev_run: bool, use_gpu: bool, ckpt_path: T.Optional[str]):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    hyperparameters = OmegaConf.load("configs/hyperparameters.yaml")
    task.connect(hyperparameters)

    # Load if a checkpoint is provided
    module = (
        LightningModule(fast_dev_run, hyperparameters, clearml_task=task)
        if not ckpt_path
        else LightningModule.load_from_checkpoint(
            ckpt_path,
            fast_dev_run=fast_dev_run,
            hyperparameters=hyperparameters,
            clearml_task=task,
            map_location="cpu",
        )
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="model-{epoch:02d}-{loss/val:.2f}",
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        verbose=True,
    )
    metrics_callback = OnTrainCallback(
        hyperparameters.val_dataset,
    )

    trainer = L.Trainer(
        max_epochs=hyperparameters.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        precision="16-mixed",
        callbacks=[metrics_callback, checkpoint_callback],
        accumulate_grad_batches=hyperparameters.accumulate_grad_batches,
        profiler=SimpleProfiler(),
        gradient_clip_val=hyperparameters.grad_norm_clip,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
    )

    if LR_FIND and not fast_dev_run:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module, min_lr=1e-6, max_lr=5e-2)

        fig = lr_finder.plot(suggest=True)
        fig.savefig("learning_rate.png")
        new_lr = lr_finder.suggestion()
        print("LEARNING RATE SUGGESTION: ", new_lr)
        assert False, "Terminate the program to avoid training"

    trainer.fit(model=module)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--ckpt", required=False, type=str, help="Checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    main(args.fast_dev_run, args.gpu, args.ckpt)
