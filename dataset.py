import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from joblib import Parallel, delayed

class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

class MVTecAT(Dataset):

    def __init__(self, root_dir, defect_name, size, transform=None, mode="train"):

        self.root_dir = Path(root_dir)
        self.defect_name = defect_name
        self.transform = transform
        self.mode = mode
        self.size = size
        
        # find test images
        if self.mode == "train":
            self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
        elif self.mode == "valid":
            self.image_names = list((self.root_dir / defect_name / "train" / "good").glob("*.png"))
            self.imgs = Parallel(n_jobs=10)(delayed(lambda file: Image.open(file).resize((size,size)).convert("RGB"))(file) for file in self.image_names)
        elif self.mode == "seg":
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
        elif self.mode == "valid_seg":
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
        else:
            #test mode
            self.image_names = list((self.root_dir / defect_name / "test").glob(str(Path("*") / "*.png")))
            
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "train":
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
                data = img[:2]
                label = img[2:]
            return data, label

        elif self.mode == "valid":
            img = self.imgs[idx].copy()
            if self.transform is not None:
                img = self.transform(img)
            return img

        elif self.mode == "valid_seg":
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size, self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

            name = filename.parts[-1]

            return img, label != "good", label, name

        elif self.mode == "seg":
            filename = self.image_names[idx]

            defect_type = filename.parts[-2]

            if defect_type != "good":
                img_name = filename.name.split('.')[0] + "_mask.png"
                ground_truth_name = (self.root_dir / self.defect_name / "ground_truth"/ defect_type / img_name )

                img = Image.open(filename)
                img = img.resize((self.size, self.size)).convert("RGB")

                ground_truth = Image.open(ground_truth_name)
                ground_truth = ground_truth.resize((self.size, self.size))
                ground_truth = np.asarray(ground_truth)

                ground_truth[ground_truth < 128] = 0
                ground_truth[ground_truth >= 128] = 1


            else:
                img = Image.open(filename)
                img = img.resize((self.size, self.size)).convert("RGB")

                ground_truth = np.zeros((self.size, self.size), dtype='int')

            if self.transform is not None:
                img = self.transform(img)
                ground_truth = torch.tensor(ground_truth)

            return img, ground_truth

        else:
            filename = self.image_names[idx]
            label = filename.parts[-2]
            img = Image.open(filename)
            img = img.resize((self.size,self.size)).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, label != "good"
