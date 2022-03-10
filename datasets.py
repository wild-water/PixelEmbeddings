from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torch
import numpy as np

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class CelebaMale(Dataset): # derm1.ipynb
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Male')]
        image = cv2.imread(im_name)    
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaSmile(Dataset): # derm2.py
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Smiling')]
        image = cv2.imread(im_name)    
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaEyeglasses(Dataset): # derm3.py
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Eyeglasses')]      
        image = cv2.imread(im_name)    
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaBald(Dataset): # derm4.py
    def __init__(self, images, attributes_list, annots):
        self.images = sorted(images)  
        self.attributes_list = attributes_list
        self.annots = annots
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        target = self.annots[im_name.split('/')[-1]][self.attributes_list.index('Bald')]     
        image = cv2.imread(im_name)    
        image = transform(image).float()
        return image, int((target + 1) // 2)

class CelebaIdentity(Dataset): # derm5.ipynb
    def __init__(self, images, identity):
        self.images = sorted(images)    
        self.identity = identity
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        ident =  self.identity[im_name.split('/')[-1]]        
        image = cv2.imread(im_name)    
        image = transform(image).float()
        return image, ident

class UTKFaceRace(Dataset): # derm6.py
    def __init__(self, images):
        self.images = sorted(images)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]    
        image = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        image = transform(image).float()
        target = int(im_name.split('_')[2])
        return image, target

class UTKFaceAge(Dataset): # derm7.py
    def __init__(self, images):
        self.images = sorted(images)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):        
        im_name = self.images[idx]    
        image = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        image = transform(image).float()
        target = int(im_name.split('/')[1].split('_')[0])
        target = np.float32(target)
        return image, target[..., None]

class UTKFaceMale(Dataset): # derm8.py
    def __init__(self, images):
        self.images = sorted(images)   
    def __len__(self):
        return len(self.images)  
    def __getitem__(self, idx):        
        im_name = self.images[idx]     
        image = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
        image = transform(image).float()
        target = int(im_name.split('_')[1])
        return image, target

class CelebaLandmarks(Dataset): # derm9.ipynb
    def __init__(self, images, landmarks):
        self.images = sorted(images)    
        self.landmarks = landmarks
    def __len__(self):
        return len(self.images)   
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        im_landmarks = self.landmarks[im_name.split('/')[-1]]        
        image = cv2.imread(im_name)    
        image = transform(image).float()
        return image, np.float32(im_landmarks)


class CelebaHair(Dataset): # derm10.ipynb
    def __init__(self, images):
        self.images = sorted(images)   
    def __len__(self):
        return len(self.images)  
    def __getitem__(self, idx):        
        im_name = self.images[idx]
        # print(im_name, idx)
        num = im_name.split('/')[-1].split('.')[0]
        num = int(num)
        segm_name = "CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(num//2000) + '/' + str(num).zfill(5) + "_hair.png"
        image = cv2.imread(im_name)  
        image = transform(image).float()
        segm_image = cv2.imread(segm_name, cv2.IMREAD_GRAYSCALE)
        if segm_image is None: # file not found == bald head
            segm_image = torch.zeros((1, 256, 256))
        segm_image = transform(segm_image).float()
        segm_image = (segm_image > 0).float()
        return image, segm_image
