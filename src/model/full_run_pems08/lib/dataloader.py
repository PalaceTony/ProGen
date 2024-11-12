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
    pos_w,
    pos_d,
    indices,
    batch_size,
    shuffle=True,
    drop_last=True,
    num_batches=None,
    num_workers=10,
    pin_memory=False,
    sampler=None,
):
    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(Y),
        torch.tensor(pos_w),
        torch.tensor(pos_d),
        torch.tensor(indices),
    )
    if sampler is not None:
        shuffle = False

    if num_batches is not None:
        batch_sampler = LimitedBatchSampler(dataset, batch_size, num_batches, shuffle)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
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
import numpy as np


from torch.utils.data import Dataset


class IndexedTensorDataset(Dataset):
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return (*tuple(tensor[index] for tensor in self.tensors), index)

    def __len__(self):
        return self.tensors[0].size(0)


def tensor_to_dataset(x, y, pos_w, pos_d, indices):
    return IndexedTensorDataset(x, y, pos_w, pos_d, indices)


def get_dataloader(
    args, local_rank, normalizer="std", single=False, use_distributed=False
):
    # Load and normalize dataset
    data = load_data(args.dataset.data_path)  # e.g., (17856, 170, 1)

    data_train_unnormalized, data_val_unnormalized, data_test_unnormalized = (
        split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    )
    test_data = data_test_unnormalized

    data, scaler = normalize_dataset(data, normalizer, args.dataset.column_wise)

    # Generate pos_w and pos_d
    points_per_day = 12 * 24
    points_per_week = points_per_day * 7
    length = len(data)
    pos_w = np.tile(np.arange(7 * points_per_day), length // points_per_week + 1)[
        :length
    ]  # Position within the week

    pos_d = np.tile(np.arange(points_per_day), length // points_per_day + 1)[
        :length
    ]  # Position within the day

    # Split dataset
    data_train, data_val, data_test = split_data_by_ratio(
        data, args.val_ratio, args.test_ratio
    )
    pos_w_train, pos_w_val, pos_w_test = split_data_by_ratio(
        pos_w, args.val_ratio, args.test_ratio
    )
    pos_d_train, pos_d_val, pos_d_test = split_data_by_ratio(
        pos_d, args.val_ratio, args.test_ratio
    )

    # Add time window
    x_tra, y_tra, pos_w_tra, pos_d_tra, indices_tra = Add_Window_Horizon(
        data_train,
        pos_w_train,
        pos_d_train,
        args.dataset.dif_model.AGCRN.T_h,
        args.dataset.dif_model.AGCRN.T_p,
        single=False,
    )

    x_val, y_val, pos_w_val, pos_d_val, indices_val = Add_Window_Horizon(
        data_val,
        pos_w_val,
        pos_d_val,
        args.dataset.dif_model.AGCRN.T_h,
        args.dataset.dif_model.AGCRN.T_p,
        single=False,
    )

    x_test, y_test, pos_w_test, pos_d_test, indices_test = Add_Window_Horizon(
        data_test,
        pos_w_test,
        pos_d_test,
        args.dataset.dif_model.AGCRN.T_h,
        args.dataset.dif_model.AGCRN.T_p,
        single=False,
    )

    # Convert to PyTorch tensors
    x_tra = torch.tensor(x_tra, dtype=torch.float32)
    y_tra = torch.tensor(y_tra, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    pos_w_tra = torch.tensor(pos_w_tra, dtype=torch.long)
    pos_w_val = torch.tensor(pos_w_val, dtype=torch.long)
    pos_w_test = torch.tensor(pos_w_test, dtype=torch.long)
    pos_d_tra = torch.tensor(pos_d_tra, dtype=torch.long)
    pos_d_val = torch.tensor(pos_d_val, dtype=torch.long)
    pos_d_test = torch.tensor(pos_d_test, dtype=torch.long)
    indices_tra = torch.tensor(indices_tra, dtype=torch.long)
    indices_val = torch.tensor(indices_val, dtype=torch.long)
    indices_test = torch.tensor(indices_test, dtype=torch.long)

    # # Check if there are any NaN values in the tensors
    # nan_check_x_tra = torch.isnan(x_tra).any()
    # nan_check_y_tra = torch.isnan(y_tra).any()
    # nan_check_x_val = torch.isnan(x_val).any()
    # nan_check_y_val = torch.isnan(y_val).any()
    # nan_check_x_test = torch.isnan(x_test).any()
    # nan_check_y_test = torch.isnan(y_test).any()

    # # Print the results
    # print("NaN in x_tra:", nan_check_x_tra)
    # print("NaN in y_tra:", nan_check_y_tra)
    # print("NaN in x_val:", nan_check_x_val)
    # print("NaN in y_val:", nan_check_y_val)
    # print("NaN in x_test:", nan_check_x_test)
    # print("NaN in y_test:", nan_check_y_test)

    # batch_sizes = [32, 64, 128, 256]  # Specify the batch sizes you want to try
    batch_sizes = [args.dataset.batch_size]  # Specify the batch sizes you want to try
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
        DistributedSampler(
            tensor_to_dataset(x_tra, y_tra, pos_w_tra, pos_d_tra, indices_tra),
            shuffle=True,
        )
        if use_distributed
        else None
    )
    val_sampler = (
        DistributedSampler(
            tensor_to_dataset(x_val, y_val, pos_w_val, pos_d_val, indices_val),
            shuffle=False,
        )
        if use_distributed
        else None
    )
    test_sampler = (
        DistributedSampler(
            tensor_to_dataset(x_test, y_test, pos_w_test, pos_d_test, indices_test),
            shuffle=args.test_batch_shuffle,
        )
        if use_distributed
        else None
    )

    # Use the `sampler` argument for all dataloaders
    train_dataloader = data_loader(
        x_tra,
        y_tra,
        pos_w_tra,
        pos_d_tra,
        indices_tra,
        args.dataset.batch_size,
        shuffle=not use_distributed,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    val_dataloader = data_loader(
        x_val,
        y_val,
        pos_w_val,
        pos_d_val,
        indices_val,
        args.dataset.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=val_sampler,
    )
    test_dataloader = data_loader(
        x_test,
        y_test,
        pos_w_test,
        pos_d_test,
        indices_test,
        args.dataset.batch_size,
        shuffle=args.test_batch_shuffle,
        drop_last=True,
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
        test_data,
    )
