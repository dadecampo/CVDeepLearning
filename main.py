import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
import glob
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as torchtrans  
from dataset import GrapeDataset, get_transform, plot_img_bbox
from detection.engine import train_one_epoch, evaluate
from detection.utils import collate_fn

if __name__ == '__main__':
    files_dir = 'dataset\Calibrated_Images'
    # use our dataset and defined transformations
    dataset = GrapeDataset(files_dir, transforms=get_transform(train=True))
    dataset_test = GrapeDataset(files_dir, transforms=get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # train test split
    test_split = 0.2
    tsize = int(len(dataset)*test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])
    print(len(dataset))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=10,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    def get_object_detection_model(num_classes):
            # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
        return model

    # train on gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2 # one class (class 0) is dedicated to the "background"

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # training for 5 epochs
    num_epochs = 5

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # the function takes the original prediction and the iou threshold.
    def apply_nms(orig_prediction, iou_thresh=0.3):
        # torchvision returns the indices of the bboxes to keep
        keep = torchvision.ops.nms(orig_prediction['boxes'].cpu(), orig_prediction['scores'].cpu(), iou_thresh)
        
        final_prediction = orig_prediction
        final_prediction['boxes'] = final_prediction['boxes'].cpu()[keep]
        final_prediction['scores'] = final_prediction['scores'].cpu()[keep]
        final_prediction['labels'] = final_prediction['labels'].cpu()[keep]
        
        return final_prediction

    # function to convert a torchtensor back to PIL image
    def torch_to_pil(img):
        return torchtrans.ToPILImage()(img).convert('RGB')

    test_dataset = GrapeDataset('dataset\Calibrated_Images', transforms= get_transform(train=True))

    # pick one image from the test set
    img, target = test_dataset[10]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]


    print('MODEL OUTPUT\n')
    nms_prediction = apply_nms(prediction, iou_thresh=0.01)

    plot_img_bbox(torch_to_pil(img), nms_prediction)