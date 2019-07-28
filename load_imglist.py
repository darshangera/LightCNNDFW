import torch.utils.data as data

from PIL import Image
import os
import os.path
import torch
import numpy as np

def default_loader(path):
    img = Image.open(path).convert('L')
    return img

def default_list_reader(fileList):
    #print(fileList)
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines()[:]:
            imgPath = line.strip().split(' ')
            #print(imgPath)
            if imgPath[2] in ['1','2','3']:
                imgList.append(imgPath)#, int(label)))
        print('There are {} pairs ..\n'.format(len(imgList)))
        return imgList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        imgpath1,imgpath2, target = self.imgList[index]
        img1 = self.loader(os.path.join(self.root,'Cropped_'+ imgpath1))
        img2 = self.loader(os.path.join(self.root,'Cropped_'+ imgpath2))
        label = int(target)
        #print(img1.size,img2.size,label)
        if label in [1,2]:  #+ve for Impersonation
            label = 1
        else:  #-ve
            if label in [3]:#-ve for Impersonation
                label = 0
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = torch.from_numpy(np.array([label],dtype=np.float32))
        #print(img1.size(),img2.size(),label)
        return img1,img2, label

    def __len__(self):
        return len(self.imgList)


