from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

app = Flask(__name__)

# 1. Load your trained model (Make sure the file name matches)
# model = torch.load('plant_model.pth', map_location=torch.device('cpu'))
# model.eval()

# 2. This helps the AI understand the image (resizing it)
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'class': 'No file uploaded'})
    
    file = request.files['file']
    img_bytes = file.read()
    
    # This is where the AI "thinks"
    # tensor = transform_image(img_bytes)
    # outputs = model(tensor)
    # _, prediction = torch.max(outputs, 1)
    
    # Placeholder result until your model is ready
    result = "Healthy Tomato Leaf" 
    
    return jsonify({'class': result})

if __name__ == '__main__':
    app.run(debug=True)
