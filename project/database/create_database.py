import os
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from ..font_identifier import load_model
import nltk
from nltk.corpus import brown
import random


class Database:
    def __init__(self, peace_and_war_path):
        nltk.download("inaugural")
        self.nltk = nltk
        with open(peace_and_war_path, 'r') as f:
            self.corpus_rus = f.read()

        self.data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(18),
            transforms.CenterCrop((18, 112)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def wrap_text(self, text, line_length=4, number_of_lines=1):
        words = text.split()
        lines = [" ".join(words[i:i+line_length]) for i in range(0, len(words), line_length)]
        assert len(lines) > number_of_lines, f"Строк должно получиться не больше {number_of_lines}"
        return "\n".join(lines[:number_of_lines])

    def random_prose_text(self, line_length=4):
        if random.random() < 0.5:
            corpus = nltk.corpus.inaugural.raw()
        else:
            corpus = self.corpus_rus
        start = random.randint(0, len(corpus) - 15000)
        text = corpus[start:start + 15000]
        return self.wrap_text(text, line_length=line_length)
    
    def create_database(self, fonts_path, dataset_path, number_of_images_per_font = 40):
        model = load_model()
        data = {}
        font_files = []
        for font_file in sorted(os.listdir(fonts_path)):
            assert (font_file.endswith('.ttf') or font_file.endswith('.ttc'), "В папке могут содержаться только шрифты")
            a = os.path.join(fonts_path, font_file)
            font_name = font_file.split('.')[0]
            font_files.append((a, font_name))
        
        for font_path, fontname in font_files:
            tensors = torch.zeros((number_of_images_per_font, 512), dtype=torch.float32)

            for j in range(number_of_images_per_font):
                img_height = random.randint(10, 33)
                font_size = max(1, img_height - 2)

                if font_path.endswith('.ttc'):
                    font = ImageFont.truetype(font_path, font_size, index=0)
                else:
                    font = ImageFont.truetype(font_path, font_size)

                font_avg_char_width = font.getbbox('x')[2]
                words_per_line = max(1, int(1500 / (font_avg_char_width * 5)))
                prose_sample = self.random_prose_text(line_length=words_per_line)

                for text in [prose_sample]:

                    img = Image.new('RGB', (800, img_height), color="white")
                    draw = ImageDraw.Draw(img)


                    font, font_size = self.choose_font(font_path, font_size, draw, img_height, text)

                    bbox = draw.textbbox((0, 0), text, font=font)
                    font_place = bbox[3] - bbox[1]
                    extra = int((img_height - font_place) / 2)

                    draw.text((0, extra - bbox[1]), text, fill="black", font=font)
                    with torch.no_grad():
                        tensors[j, :] = model(self.data_transforms(img).unsqueeze(0))
            mean_tensor = tensors.mean(0)
            data[fontname] = mean_tensor.tolist()
        with open(dataset_path+"data.json", 'w') as file:
            string = json.dumps(data)
            file.write(string)
        


    def choose_font(self, font_path, fontsize, draw, img_height, text):
        font_size = max(1, fontsize)

        while True:
            font = ImageFont.truetype(font_path, font_size, index=0) if font_path.endswith('.ttc') else ImageFont.truetype(font_path, font_size)
            bbox = draw.textbbox((0, 0), text, font=font)
            font_place = bbox[3] - bbox[1]
            extra = img_height - font_place

            if extra >= 0 or font_size == 1:
                break

            font_size -= 1

        while True:
            next_font_size = font_size + 1
            next_font = ImageFont.truetype(font_path, next_font_size, index=0) if font_path.endswith('.ttc') else ImageFont.truetype(font_path, next_font_size)
            next_bbox = draw.textbbox((0, 0), text, font=next_font)
            next_font_place = next_bbox[3] - next_bbox[1]
            next_extra = img_height - next_font_place

            if next_extra < 0:
                break

            font = next_font
            font_size = next_font_size
            bbox = next_bbox
            font_place = next_font_place
            extra = next_extra

            if extra <= 2:
                break

        return font, font_size

if __name__ == "__main__":
    database = Database(peace_and_war_path = "project/database/warandpeace.txt")
    database.create_database(fonts_path="project/fonts", dataset_path="project/database/")