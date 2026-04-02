from flask import Flask, request, send_file, jsonify
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import ViTForImageClassification
import io
import os
from flask_cors import CORS

# === CONFIG ===
NUM_CLASSES = 4
NUM_SEVERITY = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LABELS ===
CLASS_LABELS = [
    "Longitudinal Crack",  # D00
    "Transverse Crack",    # D01
    "Alligator Crack",     # D10
    "Pothole"          # D11
]
SEVERITY_LABELS = ["Low", "Medium", "High"]

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

# === Road Check Model (SmallCNN) ===
class RoadCheckModel(nn.Module):
    def __init__(self, num_classes=2):
        super(RoadCheckModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  # 224 // 8 = 28
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Model loading with fallback ---
def load_models():
    global road_model, damage_model
    try:
        road_classifier_path = os.path.join(os.path.dirname(__file__), "road_classifier.pth")
        if os.path.exists(road_classifier_path):
            road_model = RoadCheckModel().to(device)
            road_model.load_state_dict(torch.load(road_classifier_path, map_location=device))
            road_model.eval()
            print("✅ Road classifier loaded")
        else:
            print("⚠️ Road classifier missing, using mock mode")
            road_model = None
            
        vit_model_path = os.path.join(os.path.dirname(__file__), "best_damage_vit_model (2).pth")
        if os.path.exists(vit_model_path):
            damage_model = DamageViTModel(num_classes=NUM_CLASSES, num_severity=NUM_SEVERITY).to(device)
            damage_model.load_state_dict(torch.load(vit_model_path, map_location=device))
            damage_model.eval()
            print("✅ ViT damage model loaded")
        else:
            print("⚠️ ViT damage model missing, using mock mode")
            damage_model = None
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        road_model = None
        damage_model = None

load_models()

# === ROUTES ===

@app.route("/analyze/annotate", methods=["POST"])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    original_image = image.copy()

    # --- Step 1: Road Check ---
    if road_model:
        road_tensor = road_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            road_output = road_model(road_tensor)
            road_pred = torch.argmax(road_output, dim=1).item()
        if road_pred == 0:
            return jsonify({"error": "Image is not a road photo."}), 400

    # --- Step 2: Damage Prediction ---
    if damage_model:
        image_tensor = damage_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = damage_model(image_tensor)
            predicted_class = output['class_logits'].argmax(dim=1).item()
            severity_class = output['severity_logits'].argmax(dim=1).item()
            bbox = output['bbox_pred'][0].tolist()
    else:
        # MOCK DATA
        import random
        predicted_class = random.randint(0, len(CLASS_LABELS) - 1)
        severity_class = random.randint(0, len(SEVERITY_LABELS) - 1)
        bbox = [0.5, 0.5, 0.1, 0.1] # [cx, cy, bw, bh]

    # --- Draw bounding box ---
    w_img, h_img = original_image.size
    MODEL_SIZE = 224

    cx, cy, bw, bh = bbox
    # Cap width and height to match "Low" severity (area < 1%)
    bw = min(bw, 0.1) 
    bh = min(bh, 0.1)
    x1 = (cx - bw / 2) * MODEL_SIZE
    y1 = (cy - bh / 2) * MODEL_SIZE
    x2 = (cx + bw / 2) * MODEL_SIZE
    y2 = (cy + bh / 2) * MODEL_SIZE

    scale_x = w_img / MODEL_SIZE
    scale_y = h_img / MODEL_SIZE
    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    
    x1, x2 = sorted([max(0, x1), min(w_img, x2)])
    y1, y2 = sorted([max(0, y1), min(h_img, y2)])

    draw = ImageDraw.Draw(original_image)
    try:
        font = ImageFont.truetype("arial.ttf", size=30)
    except IOError:
        font = ImageFont.load_default()

    class_label = CLASS_LABELS[predicted_class]
    severity_label = SEVERITY_LABELS[severity_class]
    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    draw.text((x1, max(0, y1 - 30)), f"{class_label} ({severity_label})", fill="red", font=font)

    buffer = io.BytesIO()
    original_image.save(buffer, format="PNG")
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


@app.route("/analyze/json", methods=["POST"])
def analyze_json():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")

    # --- Step 1: Road Check ---
    if road_model:
        road_tensor = road_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            road_output = road_model(road_tensor)
            road_pred = torch.argmax(road_output, dim=1).item()
        if road_pred == 0:
            return jsonify({"error": "Image is not a road photo."}), 400

    # --- Step 2: Damage Prediction ---
    if damage_model:
        image_tensor = damage_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = damage_model(image_tensor)
            predicted_class = output['class_logits'].argmax(dim=1).item()
            severity_class = output['severity_logits'].argmax(dim=1).item()
            bbox = output['bbox_pred'][0].tolist()
    else:
        # MOCK DATA
        import random
        predicted_class = random.randint(0, len(CLASS_LABELS) - 1)
        severity_class = random.randint(0, len(SEVERITY_LABELS) - 1)
        bbox = [0.5, 0.5, 0.1, 0.1] # [cx, cy, bw, bh]

    # --- Convert bounding box to pixel coordinates ---
    w_img, h_img = image.size
    MODEL_SIZE = 224
    cx, cy, bw, bh = bbox
    bw = min(bw, 0.1)
    bh = min(bh, 0.1)
    x1 = (cx - bw / 2) * MODEL_SIZE
    y1 = (cy - bh / 2) * MODEL_SIZE
    x2 = (cx + bw / 2) * MODEL_SIZE
    y2 = (cy + bh / 2) * MODEL_SIZE

    scale_x = w_img / MODEL_SIZE
    scale_y = h_img / MODEL_SIZE
    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
    x1, x2 = sorted([max(0, x1), min(w_img, x2)])
    y1, y2 = sorted([max(0, y1), min(h_img, y2)])

    class_label = CLASS_LABELS[predicted_class]
    severity_label = SEVERITY_LABELS[severity_class]
    summary = f"A road damage of type '{class_label}' was detected with '{severity_label}' severity."

    return jsonify({
        "type": class_label,
        "severity": severity_label,
        "summary": summary,
        "bbox": [x1, y1, x2, y2]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)