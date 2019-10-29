from torch.utils.data import Dataset
import torch

class QADataset(Dataset):
    """
    This class forms our question-answer dataset! It is used in order to
    organize our data into batches and train our models.
    """
    def __init__(self,inputs,lengths_inputs,targets,masks_targets,max_trg_len):
        self.inputs = inputs
        self.lengths_inputs = lengths_inputs
        self.targets = targets
        self.masks_targets = masks_targets
        self.max_len = max_trg_len

    def __getitem__(self, item):
        return (torch.FloatTensor(self.inputs[item]), self.lengths_inputs[item],
                torch.FloatTensor(self.targets[item]), self.masks_targets[
                    item])

    def __len__(self):
        return len(self.inputs)

