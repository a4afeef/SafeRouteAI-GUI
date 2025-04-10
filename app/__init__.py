from flask import Flask
from flask_bcrypt import Bcrypt
from app.config import Keys
from app.main.routes import main

bcrypt = Bcrypt()

def create_app(config_class=Keys):
    project = Flask(__name__)
    project.config.from_object(Keys)
    bcrypt.init_app(project)
    project.register_blueprint(main)

    return project