import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io

class SubCharCNNClassifier(nn.Module):
    def __init__(self):
        super(SubCharCNNClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=30, stride=1, padding=14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=30, stride=1, padding=14)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, 8)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))  
        x = self.pool(self.relu(self.conv2(x)))  

        x = x.view(x.size(0), -1) 

        x = self.relu(self.fc1(x)) 
        x = self.fc2(x)

        return x
    
def load_model(model_path='project\exp dataset_vrbl sub_char_cnn_cnn.pt'):
    model = SubCharCNNClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.ToTensor()
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def get_image_vector(model, image_bytes):
    input_tensor = preprocess_image(image_bytes)
        
    with torch.no_grad():
        vector = model(input_tensor)
        
    return vector.squeeze().numpy().tolist()

