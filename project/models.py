from project import db

class ImageVector(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    image_data = db.Column(db.Text, nullable=False) 
    vector_data = db.Column(db.Text) 
    
