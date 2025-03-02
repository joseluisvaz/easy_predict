import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data_utils.waymo_dataset import (
    AgentCentricDataset,
    ScenarioDataset,
    collate_waymo_scenario,
    collate_waymo_stack,
)


class AgentCentricDataModule(L.LightningDataModule):
    def __init__(
        self, data_module_config: DictConfig, fast_dev_run: bool = False
    ) -> None:
        super().__init__()

        self.config = data_module_config
        self.fast_dev_run = fast_dev_run

    def setup(self, stage: str) -> None:
        self.train_dataset = AgentCentricDataset(self.config.train_dataset)
        self.val_dataset = ScenarioDataset(self.config.val_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo_stack,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            persistent_workers=True if not self.fast_dev_run else False,
            pin_memory=False,
            drop_last=True,
            prefetch_factor=4 if not self.fast_dev_run else None,
            collate_fn=collate_waymo_scenario,
        )

    def visualization_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_waymo_scenario,
        )
