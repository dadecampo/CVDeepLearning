import os
import cv2
import numpy as np
import torch
from PIL import Image
import glob
import torchvision.transforms as T
import torchvision

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2

from albumentations.augmentations.geometric.transforms import HorizontalFlip
from albumentations.augmentations.geometric.rotate import Rotate
from albumentations.augmentations.crops.transforms import BBoxSafeRandomCrop
from albumentations.augmentations.transforms import ColorJitter
from albumentations.augmentations.geometric.resize import Resize
from albumentations.augmentations.geometric.resize import RandomScale



import random

def get_transform(train):
  if train:
    return Compose(
      [
        HorizontalFlip(p=0.5),
        Rotate(limit=20, p=0.5),
        RandomScale(scale_limit=0.4, interpolation = cv2.INTER_LINEAR, p=0.5),
        BBoxSafeRandomCrop(p=0.5),
        ColorJitter(random.uniform(0,0.2), random.uniform(0,0.2), random.uniform(0,0.2), random.uniform(0,0.2), p=0.5),
        ToTensorV2(p=1.0) 
      ],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )
  else:
    return Compose(
      [
      ToTensorV2(p=1.0)
      ],
      bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
    )

class GrapeDataset(torch.utils.data.Dataset):
    def __init__(self, root, width, height, transforms = None):
        self.transforms = transforms
        self.root = root

        self.width = width
        self.height = height

        self.imgs=[]
        boxes_files=[]
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith(".txt") and not file.startswith("_counting"):
                    boxes_files.append(os.path.join(root, file))
                if file.endswith(".jpg") or file.endswith(".JPG")  :
                    self.imgs.append(os.path.join(root, file))
                if file.startswith("_counting"):
                   self.conting_file = os.path.join(root, "_counting.txt")
        self.imgs = sorted(self.imgs)
        boxes_files = sorted(boxes_files)
        self.boxes = boxes_files

    def __getitem__(self, idx):
        # get image and boxes
        img_path = self.imgs[idx]
        box_path = self.boxes[idx]
        # image elaboration
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)
        img/=255.0

        height, width, _ = img.shape
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        labels=[]
        boxes=[]
        with open(box_path) as f:
            for line in f:
                labels.append(1)

                elems=[float(x) for x in line.split(' ')]

                x_center = int(elems[1] * width)
                y_center = int(elems[2] * height)
                box_wt = int(elems[3] * width)
                box_ht = int(elems[4] * height)

                x_min = max(0.0, x_center - box_wt/2)
                x_max  = min(width, x_center + box_wt/2)
                y_min = max(0.0, y_center - box_ht/2)
                y_max  = min(height, y_center + box_ht/2)

                   

                boxes.append([x_min, y_min, x_max, y_max])

        iscrowd = None
        if len(boxes)!=0:
          boxes = torch.as_tensor(boxes, dtype=torch.float32)
          labels= torch.as_tensor(labels, dtype=torch.int64)
          iscrowd= torch.zeros((boxes.shape[0],),dtype=torch.int64)
          area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        else:
          area = torch.tensor(0)
          boxes.append([0, 0, 1, 1])
          boxes = torch.as_tensor(boxes, dtype=torch.float32)
          labels= torch.as_tensor(labels, dtype=torch.int64)
          labels = labels.unsqueeze(0)
          iscrowd = torch.zeros((boxes.shape[0],),dtype=torch.int64)


        target={}
        target["boxes"] = boxes
        target["area"] = area
        target["labels"]= labels
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image = img,
                                bboxes = target['boxes'],
                                labels = labels)
            img = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img, target

    def __len__(self):
        return len(self.imgs)
    
def plot_img_bbox(img, target, prediction):
  # plot the image and bboxes
  # Bounding boxes are defined as follows: x-min y-min width height
  fig, ax = plt.subplots(2,1)
  fig.set_size_inches(5,5)
  img=img.permute(1,2,0)
  ax[0].imshow(img)
  ax[1].imshow(img)

  for box in (target['boxes']):
    x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
    rect = patches.Rectangle(
      (x, y),
      width, height,
      linewidth = 2,
      edgecolor = 'r',
      facecolor = 'none'
    )
    # Draw the bounding box on top of the image
    ax[0].add_patch(rect)

  for box in (prediction['boxes']):
    x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
    rect = patches.Rectangle(
      (x, y),
      width, height,
      linewidth = 2,
      edgecolor = 'r',
      facecolor = 'none'
    )
    # Draw the bounding box on top of the image
    ax[1].add_patch(rect)
  plt.show()
    
