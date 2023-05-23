import random
from dataset import GrapeDataset, get_transform, plot_img_bbox


dataset_train = GrapeDataset('dataset\Calibrated_Images', get_transform(train=True))
dataset_test = GrapeDataset('dataset\Calibrated_Images', get_transform(train=False))
print("Number of Dataset Images: ", len(dataset_train))
n = random.randint(0,len(dataset_test)-1)
img, target, img_path = dataset_train[n]
plot_img_bbox(img, target)
img, target, img_path = dataset_test[n]
print(img_path)
plot_img_bbox(img, target)