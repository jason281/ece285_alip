from torch.utils.data.dataset import Dataset
import cv2
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

class DRLoader(Dataset):
    def __init__(self, input_dir, target_dir, in_transform=None, target_transform=None):
        
        assert os.path.exists(input_dir), input_dir+' not exists'
        assert os.path.exists(target_dir), target_dir+' not exists'
        self.in_transform = in_transform
        self.target_transform = target_transform
            
        self.classes = sorted(os.listdir(input_dir))
        im_path, target_path, label, classes = [], [], [], []
        for index,c in enumerate(tqdm(sorted(os.listdir(input_dir)))):
            for obj in sorted(os.listdir(input_dir+'/'+c)):
                ann, ext = os.path.splitext(obj)[0], os.path.splitext(obj)[1]
                if ext not in ['.jpeg','.png']:
                    continue
                im_path.append(os.path.join(input_dir,c,obj))
                target_path.append(os.path.join(target_dir,c,obj))
                classes.append(c)
                label.append(int(ann))
                
        self.images, self.targets, self.labels, self.c = np.array(im_path), np.array(target_path), np.array(label), np.array(classes)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        
        image = Image.open(self.images[idx])
        target = Image.open(self.targets[idx])
        label = self.labels[idx]
        classes = self.c[idx]
        if self.in_transform:
            image = self.in_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target, label,classes
    