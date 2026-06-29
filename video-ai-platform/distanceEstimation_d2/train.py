import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model_utils import estModel, load_Data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, help="batch size")
parser.add_argument('--size', type=int, help="image size")
parser.add_argument('--epoch', type=int, help="epochs")
parser.add_argument('--yaml', type=str, help="yaml path")
parser.add_argument('--workers', type=int, help="cpu workers")
args = parser.parse_args()


BATCH = args.batch if args.batch else 1
SIZE  = args.size 
EPOCHS = args.epoch if args.epoch else 1
WORKERS =  args.workers if args.workers else 2
YAML_PATH = args.yaml
assert YAML_PATH != None, "Provide data.yaml path. Use arg --yaml <path>"

# Load Data
train_data = load_Data(YAML_PATH, 'train', SIZE)
valid_data = load_Data(YAML_PATH, 'valid', SIZE)

#train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, num_workers=WORKERS, collate_fn=lambda batch: tuple(zip(*batch)))
#valid_dataloader = DataLoader(valid_data, batch_size=BATCH, shuffle=False, num_workers=WORKERS, collate_fn=lambda batch: tuple(zip(*batch)))

train_dataloader = DataLoader(train_data, batch_size=BATCH, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
valid_dataloader = DataLoader(valid_data, batch_size=BATCH, shuffle=False,  collate_fn=lambda batch: tuple(zip(*batch)))

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = estModel(num_classes=train_data.nc + 1).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Training Loop
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
best_map = 0.0
for i in range(EPOCHS):
    print(f"Epoch: {i+1}")
    model.train()
    for batch_data in train_dataloader:
        inputs, targets = batch_data
        inputs = [{k: v.to(device) for k, v in t.items()} for t in inputs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(inputs, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"[Training] -- total_loss: {losses:.6f} dist_loss: {loss_dict['loss_distance']:.6f} cls_loss: {loss_dict['loss_classifier']:.6f} box_loss: {loss_dict['loss_box_reg']:.6f} obj_loss: {loss_dict['loss_objectness']:.6f} loss_rpn_box_reg: {loss_dict['loss_rpn_box_reg']:.6f}")
    model.eval()
    with torch.no_grad():
        for batch_data in valid_dataloader:
            inputs, targets = batch_data
            inputs = [{k: v.to(device) for k, v in t.items()} for t in inputs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(inputs) 
            
            preds = [
                {
                    "boxes": o["boxes"].detach().cpu(),
                    "scores": o["scores"].detach().cpu(),
                    "labels": o["labels"].detach().cpu()
                }
                for o in outputs
            ]
            gts = [
                {
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu()
                }
                for t in targets
            ]
            metric.update(preds, gts)
    
    results = metric.compute()
    print(f"[Valid] -- map_50: {results['map_50'].item():.4f} | map_50:95: {results['map'].item():.4f}")

    if (results['map'].item() > best_map):
        best_map = results['map'].item()
        torch.save(model.state_dict().copy(), 'best.pth')
    torch.save(model.state_dict().copy(), 'last.pth')

    scheduler.step(results['map'].item())

map_per_class = {train_data.int_cls[i+1]: round(map_per_class, 4) for i, map_per_class in enumerate(results['map_per_class'].tolist())}
print(f" map_per_class: {map_per_class}")
