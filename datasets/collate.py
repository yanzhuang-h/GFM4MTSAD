# Define collate function for GNN batch training
import torch
from torch_geometric.data import Batch


def collate_fn(original_batch):
    inputs_batch = []
    labels_batch = []
    for inputs, labels in original_batch:
        inputs_batch.append(inputs)
        labels_batch.append(torch.from_numpy(labels))
    inputs_batch = Batch.from_data_list(inputs_batch)
    labels_batch = torch.stack(labels_batch, dim=0)

    return inputs_batch, labels_batch
