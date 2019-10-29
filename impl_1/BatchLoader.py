import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


class Batchloader:

    """
    data must be 3-d list or 3d numpy array
    1d: utterances
    2d: timestep - sequence length
    3d: feature size
    """
    def torch_train_val_split(self, dataset, batch_train_size, batch_eval_size,
                              val_size=.2, shuffle=True, seed=42):

        print("\n>>>Splitting to train-val sets...")

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        val_split = int(np.floor(val_size * dataset_size))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices = indices[val_split:]
        val_indices = indices[:val_split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset,
                                  batch_size=batch_train_size,
                                  sampler=train_sampler)
        val_loader = DataLoader(dataset,
                                batch_size=batch_eval_size,
                                sampler=val_sampler)

        return train_loader, val_loader
