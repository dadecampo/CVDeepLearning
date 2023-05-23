import random
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from dataset import GrapeDataset, get_transform, plot_img_bbox


dataset = GrapeDataset('dataset\Calibrated_Images', get_transform(train=True))
print("Number of Dataset Images: ", len(dataset))
img, target = dataset[random.randint(0,len(dataset)-1)]
plot_img_bbox(img, target)

def get_object_detection_model(num_classes):
  # load a model pre-trained pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
  return model

