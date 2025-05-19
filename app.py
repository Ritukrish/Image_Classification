from flask import Flask, request, jsonify, render_template
from flask import Flask, request, render_template
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Step 1: Recreate the model architecture
model = models.vit_b_16(weights=None)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, 12)  # Assuming 12 classes

# Step 2: Load state_dict from model.pkl
model.load_state_dict(torch.load('model.pkl', map_location=torch.device('cpu')))

# Step 3: Set to evaluation mode
model.eval()

# Class index to label mapping
idx_to_class = {
    0: '(BT) Body Tissue or Organ',
    1: '(GE) Glass equipment-packaging 551',
    2: '(ME) Metal equipment -packaging',
    3: '(OW) Organic wastes',
    4: '(PE) Plastic equipment-packaging',
    5: '(PP) Paper equipment-packaging',
    6: '(SN) Syringe needles',
    7: 'Gauze',
    8: 'Gloves',
    9: 'Mask',
    10: 'Syringe',
    11: 'Tweezers'
}

# Define image transformation (match what you used in training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # mean
                         [0.229, 0.224, 0.225])  # std
])
def generate_disposal_technique(label):
    suggestions = {
        "(BT) Body Tissue or Organ":
            "Place in yellow biohazard bags. Incinerate at 1100°C or deep burial as per BMW rules.",
        "(GE) Glass equipment-packaging 551":
            "Use puncture-proof containers. Disinfect with 1% NaOCl, then send for recycling.",
        "(ME) Metal equipment -packaging":
            "Clean and autoclave. Shred and recycle via authorized agencies.",
        "(OW) Organic wastes":
            "Compost biodegradable waste after disinfection. Avoid chemical treatment.",
        "(PE) Plastic equipment-packaging":
            "Rinse with 1% sodium hypochlorite. Shred and recycle using red bag protocol.",
        "(PP) Paper equipment-packaging":
            "Dry and incinerate or send for recycling after autoclaving.",
        "(SN) Syringe needles":
            "Cut needles using a needle cutter. Disinfect with NaOCl and incinerate.",
        "Gauze":
            "Soak in 1% sodium hypochlorite for 30 minutes, then autoclave and shred.",
        "Gloves":
            "Disinfect, shred, and dispose in red category waste bins.",
        "Mask":
            "Treat with 1% NaOCl, then autoclave and incinerate safely.",
        "Syringe":
            "Remove needle, disinfect syringe, and shred for safe disposal.",
        "Tweezers":
            "Autoclave and reuse if possible. If broken, disinfect and recycle as metal scrap.",
    }
    return suggestions.get(label, f"No standardized disposal found for '{label}'. Follow biomedical safety protocols.")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "Empty filename", 400

    image = Image.open(file.stream).convert('RGB')
    img_tensor = image_transforms(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        class_index = predicted.item()
        class_label = idx_to_class[class_index]
        disposal = generate_disposal_technique(class_label)

    return jsonify({
        'prediction': class_label,
        'disposal': disposal
    })


if __name__ == '__main__':
    app.run(debug=True)
