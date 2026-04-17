import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
import io
import os

class SimCLRModel(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        base_model = models.resnet18(weights=None) 
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = base_model.fc.in_features
        base_model.fc = nn.Identity()  
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return z
    

class ModelClassifier(nn.Module):
    def __init__(self, output_dim=120):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def load_model(model_path='project/contrastive_method/contrastive_model.pt'):
    encoder = SimCLRModel()
    encoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    encoder.eval()
    return encoder

def load_classifier(model_path='project/contrastive_method/classifier_model.pt'):
    classifier = ModelClassifier()
    classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    classifier.eval()
    return classifier

def crop_image(image, crop_size=40, step=40, max_crops=300):
    width, height = image.size
    crops = []

    os.makedirs("debug", exist_ok=True)

    for filename in os.listdir("debug"):
       file_path = os.path.join("debug", filename)
       os.remove(file_path)

    transform = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.ToTensor()
    ])

    idx = 0
    for y in range(0, height - crop_size + 1, step):
        for x in range(0, width - crop_size + 1, step):
            crop = image.crop((x, y, x + crop_size, y + crop_size))

            crop.save(f"debug/crop_{idx}.png")

            crops.append(transform(crop))
            idx += 1

            if len(crops) >= max_crops:
                return crops

    return crops



def get_image_vector(encoder, classifier, crop):

    input = crop.unsqueeze(0)

    with torch.no_grad():
        feats = encoder(input)
        logits = classifier(feats)
        vector = torch.sigmoid(logits.squeeze(0))
    # print(vector)

    return vector.numpy()

def get_mean_vector(encoder, classifier, image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    crops = crop_image(image)

    vectors = []

    for crop in crops:
        vec = get_image_vector(encoder, classifier, crop)
        vectors.append(vec)

    mean_vector = np.mean(vectors, axis=0)
    print(f"Средний вектор {mean_vector}")

    return mean_vector

