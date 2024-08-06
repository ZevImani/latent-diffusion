import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class lartpcBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {#"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        ## image = Image.open(example["file_path_"])
        image = Image.open(example["file_path_"]).convert('L') ## open as grayscale
        ## if not image.mode == "RGB":
        ##     image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        # print("img info:", img.shape, img.min(), img.max()) = (512, 512, 1) 0 255 

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = np.expand_dims(image, -1) ## add single channel 
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    
## example is dict with relative file path, file path, and image 


class lartpcTrain(lartpcBase):
    def __init__(self, **kwargs):
        # print(**kwargs)
        print(f' Kwargs: {kwargs}' )
        super().__init__(txt_file="/n/home11/zimani/latent-diffusion/train_images.txt", data_root="/n/home11/zimani/data_lartpc_512/train_images", **kwargs)


class lartpcValidation(lartpcBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="/n/home11/zimani/latent-diffusion/test_images.txt", data_root="/n/home11/zimani/data_lartpc_512/test_images",
                         flip_p=flip_p, **kwargs)
