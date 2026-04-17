from flask import Flask

app = Flask(__name__)

app.config['SECRET_KEY'] = '910273b92f48dd35daa0'


from project import routes