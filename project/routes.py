from flask import render_template, request
from project import db, app
import base64
from project.contrastive_models import load_model, load_classifier, get_mean_vector
from project.find_fonts import find_font

encoder = load_model()
classifier = load_classifier()

@app.route('/')
def get_vector():
  return render_template('get_vector.html', fonts=[], filename=None)

@app.route('/upload', methods=['POST'])
def upload_file():
  file = request.files['photo']
  if file: 
        image_bytes = file.read()

        vector = get_mean_vector(encoder, classifier, image_bytes)

        similar_fonts = find_font(vector, top=10)

        # new_image = ImageVector(
        #         filename=secure_filename(file.filename),
        #         image_data=image_b64,
        #         vector_data=str(vector)
        #     )
        
        # db.session.add(new_image)
        # db.session.commit()
        
        return render_template('get_vector.html', 
                            #  vector=vector,
                             filename=file.filename,
                             fonts=similar_fonts)




