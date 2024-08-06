import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch 


class lartpcBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 num_batches,
                 events_per_batch=128,
                 size=None,
                 flip_p=0.5):
        self.data_root = data_root
        self.num_batches = num_batches
        self.events_per_batch = events_per_batch
        self._length = self.num_batches * self.events_per_batch
        self.flip_p = flip_p
        self.size = size # unused? 
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        ## Size and interpolation only used for resizing images (?)

        # self.labels = {
        #     "relative_file_path_": [l for l in self.image_paths],
        #     "file_path_": [os.path.join(self.data_root, l)
        #                    for l in self.image_paths],
        # }

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        ## Load appropriate batch file 
        batch_idx = i // self.events_per_batch
        batch_file = os.path.join(self.data_root, f'protons64_{batch_idx}.npy')
        batch_data = np.load(batch_file)

        ## Get specific event 
        event_idx = i % self.events_per_batch
        event = batch_data[event_idx]

        ## Add single channel 
        event = np.expand_dims(event, -1)

        ## Probabilistic flip image - not work with tensor (TODO)
        # event = self.flip(event)

        ## Save as dictionary 
        example = {} 
        example["image"] = event.astype(np.float32)

        return example 


class protons64Train(lartpcBase):
    def __init__(self, **kwargs):
        # print(**kwargs)
        print(f' Kwargs: {kwargs}' )
        super().__init__(txt_file=None, data_root="/n/home11/zimani/datasets/protons64/train", **kwargs)


class protons64Validation(lartpcBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file=None, data_root="/n/home11/zimani/datasets/protons64/test",
                         flip_p=flip_p, **kwargs)
