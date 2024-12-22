from argparse import ArgumentParser, Namespace
from waymo_loader.dataloaders import WaymoH5Dataset, collate_waymo
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 256
NUM_WORKERS = 16


def _parse_arguments() -> Namespace:
    parser = ArgumentParser(allow_abbrev=True)
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the h5 file to process"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output h5 file with the processed data"
    )
    return parser.parse_args()


def main(data_dir: str, out: str):

    dataset = WaymoH5Dataset(data_dir, train_with_tracks_to_predict=True)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        persistent_workers=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=4,
        collate_fn=collate_waymo,
    )
    
    for batch in tqdm(dataloader):
        continue
    

if __name__ == "__main__":
    args = _parse_arguments()
    main(args.data_dir, args.out)
