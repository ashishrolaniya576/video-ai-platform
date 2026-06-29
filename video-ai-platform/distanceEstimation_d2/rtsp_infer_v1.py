import os
import yaml 
import numpy as np
import cv2
import torch

from model_utils import estModel

rtsp_URL = "rtsp://10.10.15.30:4554/1069"
model_PATH = "best.pth"
yaml_PATH = "data.yaml" # Use same xml as used for training
fov = 0.1 # FOV value of this image
image_size  = 640 # Ensure this value is same as used in training
threshold = 0.7 # show objects with > 70% confidence scores

def read_YAML(path):
    assert os.path.exists(path), f"File doesn't exist: {path}"
    # Read data.yaml
    with open(path, 'r') as file:
        contents = yaml.safe_load(file)
    return contents['nc'], contents['names']

def process_IMG(frame, size=None):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR to RGB

    if (size):
        h, w, _ = image.shape
        aspect_ratio = h / w
        n_w = int(size)
        n_h = int(aspect_ratio * n_w)
                    
        image = cv2.resize(image, (n_w, n_h), interpolation=cv2.INTER_AREA)

    return torch.tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32) / 255.0 

def predictions(input, model, device):
    with torch.no_grad():
        output = model([{k: v.to(device) for k, v in input.items()}])[0]
    
    # Filter predictions by score threshold
    boxes = output['boxes'].cpu()
    labels = output['labels'].cpu()
    scores = output['scores'].cpu()
    distances = output['distances'].cpu()

    selected_indices = scores > threshold
    boxes = boxes[selected_indices]
    labels = labels[selected_indices]
    scores = scores[selected_indices]
    distances = distances[selected_indices]

    return boxes, labels, scores, distances

def visualize(frame, preds, int_cls):
    frame = cv2.cvtColor(np.transpose(image, (1, 2, 0)).numpy() * 255, cv2.COLOR_RGB2BGR)
    frame = frame.astype(np.uint8)

    boxes, labels, scores, distances = preds
    for box, label, score, distance in zip(boxes, labels, scores, distances):
        x1, y1, x2, y2 = box

        # Draw rectangle
        cv2.rectangle(frame, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), (0, 255, 0), 2)

        # Put label text
        class_name = int_cls[(label.item())]
        text = f"{class_name} {score.item():.2f} {distance.item():.2f}"
        cv2.putText(frame, text, (int(x1.item()), int(y1.item()) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("RTSP Stream", frame)

# Read YAML file
num_classes, class_names  = read_YAML(yaml_PATH)

# create class to int and vice versa dictionary [to be used for visualization]
cls_int = {c: i+1 for i, c in enumerate(class_names)}
cls_int['background'] = 0
int_cls = { values:keys for keys, values in cls_int.items()}

# Init model
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = estModel(num_classes=num_classes + 1).to(device)
model.load_state_dict(torch.load(model_PATH, map_location=torch.device('cpu')))
#model.load_state_dict(torch.load(model_PATH))
model.eval()

# Initialize video capture
cap = cv2.VideoCapture(rtsp_URL)
if not cap.isOpened():
    print("Error: Cannot open RTSP stream")
    exit()

cnt = 0
SKIP_FRAMES = 10
# Capture and process frames
while True:
    # Read a frame from the RTSP stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break
    
    cnt += 1
    if(cnt % SKIP_FRAMES != 0):
        continue
    cnt = cnt % SKIP_FRAMES

    # Read an image
    image = process_IMG(frame, image_size)

    # Create input dictionary with image and fov
    inputs = {'image': image, 'fov': torch.tensor(fov, dtype=torch.float32)} # FOV is used here

    # Inference on single frame
    preds = predictions(inputs, model, device)

    # Visualize predictions
    visualize(image, preds, int_cls)
    
    # Use q to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

