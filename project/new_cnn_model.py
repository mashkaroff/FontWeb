import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import json

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 79)
model.load_state_dict(torch.load('project/font_identifier_model.pth'))
model.fc = nn.Identity()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Convert images to grayscale with 3 channels
        transforms.RandomCrop((224, 224)), # Resize images to the expected input size of the model
        transforms.ToTensor(), # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize with ImageNet stats
    ])  
    
    image_tensor = data_transforms(image).unsqueeze(0)
    
    return image_tensor

def get_fonts(image_tensor):
    model.eval()
    with torch.no_grad():
        vector = model(image_tensor)
        vector = vector.squeeze()

    # print(vector)

    top = []
    with open("project/data.json", "r") as file:
        data = json.load(file)
        for key in data.keys():
            data[key]= torch.tensor(data[key])
            top.append({
                "name": key, 
                "sim": F.cosine_similarity(vector, data[key], dim=0).item(),
                "image": None
                })

    sorted_top = sorted(top, key=lambda x: x["sim"], reverse=True)
    # print(top)
    # print(sorted_top[:5])

    return sorted_top