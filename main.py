import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
import glob
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FastRCNNConvFCHead
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as torchtrans  
from dataset import GrapeDataset, get_transform, plot_img_bbox
from detection.engine import train_one_epoch, evaluate
from detection.utils import collate_fn
import torch
import gc
import time
from tqdm import tqdm

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 1
EPOCHS = 15
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 500

if __name__ == '__main__':
    files_dir = 'dataset\Calibrated_Images\with_Counting\Multiple_Cultivar_BBCH83_13_08_20'
    #files_dir = 'dataset\Calibrated_Images'
    # use our dataset and defined transformations
    dataset = GrapeDataset(files_dir, IMAGE_WIDTH, IMAGE_HEIGHT, transforms=get_transform(train=True))
    dataset_test = GrapeDataset(files_dir, IMAGE_WIDTH, IMAGE_HEIGHT, transforms=get_transform(train=False))

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    def get_object_detection_model(num_classes):
        import torchvision
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models.detection import retinanet_resnet50_fpn_v2
        from torchvision.models.detection import RetinaNet, RetinaNet_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights
        from torchvision.models import ResNet50_Weights, resnet101
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
        from functools import partial

        #weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )   
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        return model
    
    def get_yolo_object_detection_model(num_classes):
        # the neural network configuration
        config_path = "yolov3/cfg/yolov3.cfg"
        # the YOLO net weights file
        weights_path = "yolov3/weights/yolov3.weights"
        model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        return model
    


    num_classes = 2 # one class (class 0) is dedicated to the "background"

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(DEVICE)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.001)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    def train_epoch(model, optimizer, data_loader, device, epoch):
        model.train()

        for i, (images, targets) in (tepoch := tqdm(enumerate(data_loader), unit="batch", total=len(data_loader))):
            tepoch.set_description(f"Epoch {epoch}")
            # Step 1: send the image to the required device.
            # Images is a list of B images (where B = batch_size of the DataLoader).
            images = list(img.to(device) for img in images)
            # Step 2: send each target to the required device
            # Targets is a dictionary of metadata. each (k,v) pair is a metadata
            # required for training.
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model_time = time.time()
            loss_dict=model(images, targets)
            model_time = time.time() - model_time
            # Step 3. backward on loss.
            # Normally, you would obtain the loss from the model.forward()
            # and then just call .bacward() on it.
            # In this case, for each task, you have a different loss, due to
            # different error metrics adopted by the tasks.
            # One typical approach is to combine all the losses to one single loss,
            # and then then backward that single loss.
            # In this way you can adjust the weight of the different tasks,
            # multiplying each loss for a hyperparemeter.
            # E.G.:
            #       final_loss = loss_1 + gamma*(alpha*loss_2 + beta*loss_3)
            # In this case, we want to sum up all the losses.
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            tepoch.set_postfix(loss=losses.item())

    # training for 5 epochs

    for epoch in range(EPOCHS):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, DEVICE, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=DEVICE)

        
    evaluate(model, data_loader_test, device=DEVICE)

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
    test_dataset = GrapeDataset(files_dir, IMAGE_WIDTH, IMAGE_HEIGHT, transforms= get_transform(train=False))

    # pick one image from the test set
    img, target = test_dataset[8]
    # put the model in evaluation mode
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(DEVICE)])[0]


    print('MODEL OUTPUT\n')
    nms_prediction = apply_nms(prediction, iou_thresh=0.01)

    plot_img_bbox(img, target, nms_prediction)