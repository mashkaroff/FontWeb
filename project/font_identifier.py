import os
import io
import torch
import pytesseract
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image, ImageDraw
import json

def load_model():
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 71)
    model.load_state_dict(torch.load('project/font_identifier_model_lines.pth'))
    model.fc = nn.Identity()
    model.eval()

    return model

def get_rows(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    rows = []
    n = len(data["level"])

    for i in range(n):
        if data["level"][i] == 4:
            left = data["left"][i]
            top = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]

            if width > 200 and height > 5:
                bbox = [left, top, left + width, top + height]
                rows.append(image.crop(bbox))

    return rows

def row_to_vec(model, row_image):

    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(18),
        transforms.CenterCrop((18, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = data_transforms(row_image).unsqueeze(0)

    with torch.no_grad():
        vector = model(image_tensor).squeeze()
        # print(vector)

    return vector

def get_similar_fonts(image_bytes):

    model = load_model()
    rows = get_rows(image_bytes)

    vectors = []

    for row in rows:
        vectors.append(row_to_vec(model, row))

    mean_vector = torch.stack(vectors).mean(dim=0)
    # print(mean_vector)

    top = []

    with open("project/database/data.json", "r") as file:
        font_data = json.load(file)
        for key in font_data.keys():
            font_data[key]= torch.tensor(font_data[key])
            top.append({
                "name": key, 
                "sim": F.cosine_similarity(mean_vector, font_data[key], dim=0).item(),
                "image": None
                })

    sorted_top = sorted(top, key=lambda x: x["sim"], reverse=True)
        

    return sorted_top[:5]



# проверка выделения строк на 1 картинке
if __name__ == "__main__":
    print(os.getcwd(), "!!!!")
    
    image_path = "project/examples/cascadia_code_extralight.png"
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
    else:
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, lang="rus", output_type=pytesseract.Output.DICT)
        print(data)

        bboxes = []

        n = len(data["level"])

        for i in range(n):
            if data["level"][i] == 4:
                left = data["left"][i]
                top = data["top"][i]
                width = data["width"][i]
                height = data["height"][i]

                if width > 200 and height > 5:
                    bbox = [left, top, left + width, top + height]
                    bboxes.append(bbox)

        draw = ImageDraw.Draw(img)
        for bbox in bboxes:
            print(bbox)
            draw.rectangle(bbox, outline="red", width=2)
            crop = img.crop(bbox)
            crop.show()

        img.show()