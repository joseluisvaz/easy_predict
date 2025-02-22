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
from prediction_module import PredictionLightningModule

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("medium")


def main(fast_dev_run: bool, use_gpu: bool, ckpt_path: T.Optional[str], task: T.Optional[Task]):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    hyperparameters = OmegaConf.load("configs/hyperparameters.yaml")

    if task is not None:
        task.connect(hyperparameters)

    module = (
        PredictionLightningModule(fast_dev_run, hyperparameters, clearml_task=task)
        if not ckpt_path
        else PredictionLightningModule.load_from_checkpoint(
            ckpt_path,
            clearml_task=task,
            map_location="cpu",
        )
    )

    callbacks = [
        OnTrainCallback(hyperparameters.val_dataset),
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="model-{epoch:02d}-{loss/val:.2f}",
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            verbose=True,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=hyperparameters.max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        fast_dev_run=fast_dev_run,
        precision="16-mixed",
        callbacks=callbacks,
        accumulate_grad_batches=hyperparameters.accumulate_grad_batches,
        profiler=SimpleProfiler() if args.profile else None,
        gradient_clip_val=hyperparameters.grad_norm_clip,
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
    )

    if args.lr_find and not fast_dev_run:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(module, min_lr=2e-4, max_lr=2e-4)

        fig = lr_finder.plot(suggest=True)
        fig.savefig("learning_rate.png")
        new_lr = lr_finder.suggestion()
        print("LEARNING RATE SUGGESTION: ", new_lr)
        assert False, "Terminate the program to avoid training"

    trainer.fit(model=module, ckpt_path=ckpt_path)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument("--fast-dev-run", action="store_true", help="Fast dev run")
    parser.add_argument("--profile", action="store_true", help="Profile the training")
    parser.add_argument("--lr_find", action="store_true", help="LR find")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--ckpt", required=False, type=str, help="Checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    
    if not args.fast_dev_run:
        task = Task.init(project_name="TrajectoryPrediction", task_name="SimpleAgentPrediction MCG")
    else:
        task = None
    main(args.fast_dev_run, args.gpu, args.ckpt, task)
