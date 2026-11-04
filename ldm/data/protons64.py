import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch 


class protons(Dataset):
	def __init__(self,
				 data_root,
				 num_batches,
				 events_per_batch,
				 only_one=False,
				 v1=False,
				 mom_scale=500.0,
				 in_memory=False):
		self.data_root = data_root
		self.num_batches = num_batches
		self.events_per_batch = events_per_batch
		self._length = self.num_batches * self.events_per_batch
		self.only_one = only_one
		if self.only_one:
			self._length = 1000
		self.v1 = v1
		self.mom_scale = mom_scale

		## Load every batch file once so __getitem__ is just an index (electrons train: 4.2 GB).
		## Otherwise each event reads its whole 128-event file (~175 events/s per worker).
		## DataLoader workers are forked after this, so they share the arrays instead of copying them.
		self.in_memory = in_memory
		if self.in_memory:
			first = np.load(self._file(0))
			self.images = np.empty((self.num_batches * self.events_per_batch, *first.shape[1:]), dtype=first.dtype)
			self.moms = np.empty((self.num_batches * self.events_per_batch, 3), dtype=np.float32)
			for b in range(self.num_batches):
				sl = slice(b * self.events_per_batch, (b + 1) * self.events_per_batch)
				self.images[sl] = first if b == 0 else np.load(self._file(b))
				self.moms[sl] = np.load(self._file(b, mom=True))

	def _file(self, batch_idx, mom=False):
		prefix = 'protons64' if self.v1 else 'batch'
		suffix = '_mom' if mom else ''
		return os.path.join(self.data_root, f'{prefix}{suffix}_{batch_idx}.npy')

	def __len__(self):
		return self._length

	def __getitem__(self, i):

		## Use same event every time (testing purposes)
		if self.only_one:
			i = 0

		if self.in_memory:
			event, mom = self.images[i], self.moms[i]
		else:
			## Load appropriate batch file and corresponding momentum
			batch_idx = i // self.events_per_batch
			batch_data = np.load(self._file(batch_idx))
			mom_data = np.load(self._file(batch_idx, mom=True))

			## Get specific event
			event_idx = i % self.events_per_batch
			event, mom = batch_data[event_idx], mom_data[event_idx]

		## Normalize momentum
		mom = mom / self.mom_scale ## protons (edep v2): ~[-445, 445] MeV/c / 500 -> ~[-0.89, 0.89]

		## Add single channel 
		event = np.expand_dims(event, -1)

		## Save as dictionary 
		example = {} 
		example["image"] = event.astype(np.float32)
		example["momentum"] = mom.astype(np.float32)

		return example 



class protons64Train_v1(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64_div10/train", 
						 num_batches=1152, events_per_batch=128, v1=True, **kwargs)

class protons64Validation_v1(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64_div10/val",
						 num_batches=152, events_per_batch=128, v1=True, **kwargs)

class protons64xTrain_v1(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64x_div10/train",  
						 num_batches=155, events_per_batch=128, v1=True, **kwargs)

class protons64xValidation_v1(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64x_div10/val",
						num_batches=21, events_per_batch=128, v1=True, **kwargs)



class protons64Train(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64_div10_v2/train", 
						 num_batches=1104, events_per_batch=128, **kwargs)

class protons64Validation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64_div10_v2/val",
						 num_batches=145, events_per_batch=128, **kwargs)

class protons64xTrain(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64x_div10_v2/train",  
						 num_batches=155, events_per_batch=128, **kwargs)

class protons64xValidation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/protons64x_div10_v2/val",
						num_batches=21, events_per_batch=128, **kwargs)


# class edepProtons64Train(protons):
# 	def __init__(self, **kwargs):
# 		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64/edep_train", 
# 						 num_batches=2000, events_per_batch=128, **kwargs)

# class edepProtons64Validation(protons):
# 	def __init__(self, **kwargs):
# 		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64/edep_val",
# 						 num_batches=200, events_per_batch=128, **kwargs)

class edepProtons64Train(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64_v2/edep_train", 
						 num_batches=2000, events_per_batch=128, **kwargs)

class edepProtons64Validation(protons):
	def __init__(self, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/edep_protons64_v2/edep_val",
						 num_batches=200, events_per_batch=128, **kwargs)


## Electrons: 10-100 MeV, |p| in [10, 100] MeV/c -> mom_scale=100 gives [-1, 1]
class electrons64Train(protons):
	def __init__(self, mom_scale=100.0, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/electrons/train",
						 num_batches=2000, events_per_batch=128, mom_scale=mom_scale, **kwargs)

class electrons64Validation(protons):
	def __init__(self, mom_scale=100.0, **kwargs):
		super().__init__(data_root="/n/holystore01/LABS/iaifi_lab/Users/zimani/datasets/electrons/val",
						 num_batches=200, events_per_batch=128, mom_scale=mom_scale, **kwargs)
