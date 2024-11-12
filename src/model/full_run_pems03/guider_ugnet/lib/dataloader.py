from lib.add_window import Add_Window_Horizon
from lib.normalization import (
    NScaler,
    MinMax01Scaler,
    MinMax11Scaler,
    StandardScaler,
    ColumnMinMaxScaler,
)

import torch
import torch.utils.data
from lib.load_dataset import load_data
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler


def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == "max01":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax01 Normalization")
    elif normalizer == "max11":
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
        print("Normalize the dataset by MinMax11 Normalization")
    elif normalizer == "std":
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
        print("Normalize the dataset by Standard Normalization")
    elif normalizer == "None":
        scaler = NScaler()
        data = scaler.transform(data)
        print("Does not normalize the dataset")
    elif normalizer == "cmax":
        # column min max, to be depressed
        # note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
        print("Normalize the dataset by Column Min-Max Normalization")
    else:
        raise ValueError
    return data, scaler


def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len * test_ratio) :]
    val_data = data[
        -int(data_len * (test_ratio + val_ratio)) : -int(data_len * test_ratio)
    ]
    train_data = data[: -int(data_len * (test_ratio + val_ratio))]
    return train_data, val_data, test_data


class LimitedBatchSampler:
    def __init__(self, data_source, batch_size, num_batches, shuffle):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            sampler = RandomSampler(self.data_source)
        else:
            sampler = SequentialSampler(self.data_source)
        batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=True)

        for idx, batch in enumerate(batch_sampler):
            if idx < self.num_batches:
                yield batch

    def __len__(self):
        return self.num_batches


def data_loader(
    X,
    Y,
    batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=10,
    pin_memory=True,
    sampler=None,  # Add a sampler argument
):
    dataset = torch.utils.data.TensorDataset(X, Y)
    if sampler is not None:
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
    )
    return dataloader


from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
import logging


def tensor_to_dataset(x, y):
    return TensorDataset(x, y)


def get_dataloader(
    args, local_rank, normalizer="std", single=False, use_distributed=False
):
    # Load and normalize dataset
    data = load_data(args.data_path)  # e.g., (17856, 170, 1)
    data, scaler = normalize_dataset(data, normalizer, args.column_wise)

    # Split dataset
    data_train, data_val, data_test = split_data_by_ratio(
        data, args.val_ratio, args.test_ratio
    )

    # Add time window
    x_tra, y_tra = Add_Window_Horizon(
        data_train, args.dif_model.AGCRN.T_h, args.dif_model.AGCRN.T_p, single
    )
    x_val, y_val = Add_Window_Horizon(
        data_val, args.dif_model.AGCRN.T_h, args.dif_model.AGCRN.T_p, single
    )
    x_test, y_test = Add_Window_Horizon(
        data_test, args.dif_model.AGCRN.T_h, args.dif_model.AGCRN.T_p, single
    )

    # Convert to PyTorch tensors
    x_tra = torch.tensor(x_tra, dtype=torch.float32)
    y_tra = torch.tensor(y_tra, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # batch_sizes = [32, 64, 128, 256]  # Specify the batch sizes you want to try
    batch_sizes = [args.batch_size]  # Specify the batch sizes you want to try
    # batch_sizes = list(range(198, 220))  # Specify the batch sizes you want to try

    if args.use_distributed:
        world_size = torch.distributed.get_world_size()
        local_rank = torch.distributed.get_rank()
    else:
        world_size = 1
        local_rank = 0

    for batch_size in batch_sizes:
        # Calculate the total number of batches if the dataset were not split
        total_train_batches = len(x_tra) // batch_size
        total_val_batches = len(x_val) // batch_size
        total_test_batches = len(x_test) // batch_size

        # Calculate the number of batches each GPU processes
        batches_per_gpu_train = total_train_batches // world_size
        batches_per_gpu_val = total_val_batches // world_size
        batches_per_gpu_test = total_test_batches // world_size

        if local_rank == 0:
            logging.info(f"Batch size: {batch_size}")
            logging.info(
                f"Total train batches: {total_train_batches}, Train batches per GPU: {batches_per_gpu_train}"
            )
            logging.info(
                f"Total validation batches: {total_val_batches}, Validation batches per GPU: {batches_per_gpu_val}"
            )
            logging.info(
                f"Total test batches: {total_test_batches}, Test batches per GPU: {batches_per_gpu_test}"
            )

    # Create a DistributedSampler for all datasets if using distributed training
    train_sampler = (
        DistributedSampler(tensor_to_dataset(x_tra, y_tra), shuffle=True)
        if use_distributed
        else None
    )
    val_sampler = (
        DistributedSampler(tensor_to_dataset(x_val, y_val), shuffle=False)
        if use_distributed
        else None
    )
    test_sampler = (
        DistributedSampler(tensor_to_dataset(x_test, y_test), shuffle=False)
        if use_distributed
        else None
    )

    # Use the `sampler` argument for all dataloaders
    train_dataloader = data_loader(
        x_tra,
        y_tra,
        args.batch_size,
        shuffle=not use_distributed,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    val_dataloader = data_loader(
        x_val,
        y_val,
        args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=val_sampler,
    )
    test_dataloader = data_loader(
        x_test,
        y_test,
        args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=test_sampler,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        scaler,
        train_sampler,
        val_sampler,
        test_sampler,
    )
