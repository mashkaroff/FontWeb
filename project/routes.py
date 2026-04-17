from flask import render_template, request
from project import app
import base64
import os
# from project.contrastive_method.contrastive_models import load_model, load_classifier, get_mean_vector
# from project.contrastive_method.find_fonts import find_font
from project.font_identifier import get_similar_fonts
from project.utils import generate_font

# encoder = load_model()
# classifier = load_classifier()
FONTS_FOLDER = "project/fonts"

@app.route('/')
def get_vector():
  return render_template('get_vector.html', fonts=[], filename="")

@app.route('/upload', methods=['POST'])
def upload_file():
  file = request.files['photo']
  if file: 
        image_bytes = file.read()
        upload_image = base64.b64encode(image_bytes).decode('utf-8')

        similar_fonts = get_similar_fonts(image_bytes)
        for font_dict in similar_fonts:
           font_dict["image"] = generate_font(os.path.join(FONTS_FOLDER, font_dict["name"] + ".ttf"))

        # для модели контрастного обучения
        # vector = get_mean_vector(encoder, classifier, image_bytes)
        # similar_fonts = find_font(vector, top=10)
        
        return render_template('get_vector.html', 
                            #  vector=vector,
                             filename=file.filename,
                             fonts=similar_fonts,
                             uploaded_image=upload_image)




