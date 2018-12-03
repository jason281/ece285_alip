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
    
class Syn2Real(Dataset):
    def __init__(self, back_dir, ori_dir, syn_dir, in_transform=None, target_transform=None):
        
        assert os.path.exists(back_dir), back_dir+' not exists'
        assert os.path.exists(ori_dir), ori_dir+' not exists'
        assert os.path.exists(syn_dir), syn_dir+' not exists'
        self.in_transform = in_transform
        self.target_transform = target_transform
            
        self.classes = sorted(os.listdir(ori_dir))
        ori_im_path, back_im_path, syn_im_path, classes = [], [], [], []
        for c in tqdm(self.classes):
            for obj in sorted(os.listdir(ori_dir+'/'+c)):
                if obj[0] == '.':
                    continue
                ann, ext = os.path.splitext(obj)[0], os.path.splitext(obj)[1]
                ori_im_path.append(os.path.join(ori_dir,c,obj))
                back_im_path.append(os.path.join(back_dir,c,ann+'.png'))
                syn_im_path.append(os.path.join(syn_dir,c,ann+'.png'))
                
                classes.append(c)
                
        self.ori_im_path = np.array(ori_im_path)
        self.back_im_path = np.array(back_im_path)
        self.syn_im_path = np.array(syn_im_path)
        self.c = np.array(classes)
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        
        background = Image.open(self.back_im_path[idx])
        synthetic = Image.open(self.syn_im_path[idx])
        original = Image.open(self.ori_im_path[idx])
        
        input_image = background.paste(synthetic)
        classes = self.c[idx]
        if self.in_transform:
            image = self.in_transform(input_image)
        if self.target_transform:
            target = self.target_transform(original)

        return image, target
    