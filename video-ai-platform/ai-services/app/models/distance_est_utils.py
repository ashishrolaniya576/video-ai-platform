import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vision
from torch.utils.data import Dataset

class load_Data(Dataset):
    def __init__(self, yaml_PATH, split, size=None):
        super().__init__()
        self.split = split
        self.size = size
        self.nc, self.names, self.root_DIR = self.read_YAML(yaml_PATH)
        assert int(self.nc) == len(self.names), "Mismatch between 'nc' and the number of class names in data.yaml."
        self.cls_int = {c: i+1 for i, c in enumerate(self.names)}
        self.cls_int['background'] = 0
        self.int_cls = { values:keys for keys, values in self.cls_int.items()}
        split_PATH = os.path.join(self.root_DIR, split, 'images')
        assert os.path.exists(split_PATH), f"Path doesn't exist: {split_PATH}"
        self.data = os.listdir(split_PATH)
        print(f"{split} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def scale_IMG(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR_BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (self.size):
            h, w, _ = image.shape
            aspect_ratio = h / w
            n_w = int(self.size)
            n_h = int(aspect_ratio * n_w)         
            image = cv2.resize(image, (n_w, n_h), interpolation=cv2.INTER_AREA)
        return torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) / 255.0

    def read_YAML(self, path):
        assert os.path.exists(path), f"File doesn't exist: {path}"
        with open(path, 'r') as file:
            contents = yaml.safe_load(file)
        return contents['nc'], contents['names'], contents['path']
    
    def annotations(self, path):
        #print("path -------------------",path)
        tree = ET.parse(path)
        root = tree.getroot()
        if (self.size):
            s = root.find('size')
            w = int(s.find('width').text)
            h = int(s.find('height').text)
            # c = int(s.find('depth').text)
            aspect_ratio = h / w
            n_w = int(self.size)
            n_h = int(aspect_ratio * n_w)
            scale_x = n_w / w
            scale_y = n_h / h
        boxs = []
        dists = []
        lbls = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            distance = float(obj.find('distance').text)
            fov = float(obj.find('fov').text) ###
            assert 0.0 < distance <= 1.0, f"Distance values should be in (0, 1) range, instead got: {distance}"
            if (self.size):
                xmin = xmin * scale_x
                ymin = ymin * scale_y
                xmax = xmax * scale_x
                ymax = ymax * scale_y
            boxs.append([xmin, ymin, xmax, ymax])
            lbls.append(self.cls_int[name])
            dists.append(distance)
        targets = {'boxes' : torch.tensor(np.array(boxs, dtype=float), dtype=torch.float32),
                    'labels' : torch.tensor(lbls, dtype=torch.int64), 
                    'distances' : torch.tensor(dists, dtype=torch.float32)}
                    
        return torch.tensor(fov, dtype=torch.float32), targets ##
    
    def __getitem__(self, idx):
        img_PATH = os.path.join(self.root_DIR, self.split, 'images', self.data[idx])
        #print("img_PATH ",img_PATH)
        lbl_PATH = os.path.join(self.root_DIR, self.split, 'labels', self.data[idx].split('.')[0]+'.xml')
        #print("img_PATH  lbl_PATH",lbl_PATH,img_PATH)
        assert os.path.exists(img_PATH), f"Path doesn't exist: {img_PATH}"
        assert os.path.exists(lbl_PATH), f"Path doesn't exist: {lbl_PATH}"
        image = self.scale_IMG(img_PATH)
        fov, targets = self.annotations(lbl_PATH) ##

        inputs = {'image': image, 'fov': fov} ###
        return inputs, targets ##

class cHead(vision.models.detection.roi_heads.RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        representation_size = self.box_head.fc7.out_features
        self.distance_head = nn.Sequential(
            nn.Linear(representation_size + 1, 512),
            nn.Softplus(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, proposals, image_shapes, targets=None, fovs=None): ##
        
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            gt_distances = []
            fov_features = [] ###
            for img_idx, matched in enumerate(matched_idxs):
                num_samples = matched.numel() ###
                target_dist = targets[img_idx]["distances"]
                gt_dist = target_dist[matched]
                fov_val = torch.as_tensor(fovs[img_idx], dtype=torch.float32,      ###
                                          device=box_features.device).view(1, 1)
                gt_distances.append(gt_dist)
                fov_features.append(fov_val.expand(num_samples, 1)) ###
            gt_distances = torch.cat(gt_distances, dim=0)
            fov_features = torch.cat(fov_features, dim=0)  ###
            box_features_with_fov = torch.cat([box_features, fov_features], dim=1)
            predicted_distances = self.distance_head(box_features_with_fov).squeeze(1) ##
            distance_loss = F.smooth_l1_loss(predicted_distances, gt_distances)
            class_logits, box_regression = self.box_predictor(box_features)
            loss_classifier, loss_box_reg = vision.models.detection.roi_heads.fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_distance": distance_loss
            }
            return [], losses
        else:
            detections, _ = super().forward(features, proposals, image_shapes, targets)
            if len(detections) == 0:
                return detections, {}
            box_features = self.box_roi_pool(features, [d["boxes"] for d in detections], image_shapes)
            box_features = self.box_head(box_features)
            fov_features = [] ###
            for img_idx, d in enumerate(detections): ###
                num_boxes = d["boxes"].shape[0] ###
                fov_val = torch.as_tensor(fovs[img_idx], dtype=torch.float32, ###
                                          device=box_features.device).view(1, 1)
                fov_features.append(fov_val.expand(num_boxes, 1)) ###
            fov_features = torch.cat(fov_features, dim=0) ###
            box_features_with_fov = torch.cat([box_features, fov_features], dim=1) ###

            predicted_distances = self.distance_head(box_features_with_fov).squeeze(1) ##
            
            start = 0
            for d in detections:
                num_boxes = d["boxes"].shape[0]
                d["distances"] = predicted_distances[start:start + num_boxes]
                start += num_boxes
            return detections, {}


class estModel(vision.models.detection.FasterRCNN): ##
    def __init__(self, num_classes):
        #super().__init__()
        
        backbone = vision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', weights=None)
        anchor_generator = vision.models.detection.rpn.AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        roi_pooler = vision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )
        super().__init__(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )

        self.roi_heads = cHead(
            box_roi_pool=self.roi_heads.box_roi_pool,
            box_head=self.roi_heads.box_head,
            box_predictor=self.roi_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
        )

    def forward(self, inputs, targets=None): ##
        # return self.model(images, targets)
        images = [inp["image"] for inp in inputs]
        fovs = [inp["fov"] for inp in inputs]
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(
            features, proposals, images.image_sizes, targets, fovs=fovs
        )
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        if self.training:
            return losses
        return detections
