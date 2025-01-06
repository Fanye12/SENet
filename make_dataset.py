import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
random.seed(3407)

class make_Dataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
    
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')] #img path list
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]          #mask path list
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.size = len(self.images)      #length of dataset

        self.img_transform = transforms.Compose([                            #对图片进行预处理
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
 
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            # transforms.Resize((trainsize, trainsize), interpolation=Image.NEAREST), #用这种插值方式gt里面只有0和1
            transforms.ToTensor()])
        
    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        self.getFlip()
        image = self.flip1(image)
        gt = self.flip1(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # weit = gt
        return image, gt#, weit

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')#转为灰度图
        
    def getFlip(self):
        p1 = random.randint(0, 1)
        self.flip1 = transforms.RandomHorizontalFlip(p1)

    def __len__(self):
        return self.size

class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, gt_root, testsize):

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.img_transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def get_loader(image_root, gt_root, 
               batchsize = 16, trainsize=384, shuffle=True, num_workers=12, pin_memory=True):
    # `num_workers=0` for more stable training
    dataset = make_Dataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                #   drop_last = True,
                                  pin_memory=pin_memory)

    return data_loader
