import random
import numpy as np
from dataset import GrapeDataset, get_transform, plot_img_bbox

files_path='dataset\Calibrated_Images\with_Counting'
dataset_train = GrapeDataset(files_path, 4200, 2800, get_transform(train=True))
dataset_test = GrapeDataset(files_path, 4200, 2800,  get_transform(train=False))
print("Number of Dataset Images: ", len(dataset_train))
n = random.randint(0,len(dataset_test)-1)
img, target = dataset_train[n]
plot_img_bbox(img, target)
img, target = dataset_test[n]
plot_img_bbox(img, target)