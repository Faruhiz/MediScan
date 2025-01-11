import os

class Config:
    # DB
    SQLALCHEMY_DATABASE_URI = 'sqlite:///metadata.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload folder
    UPLOAD_FOLDER = './uploads'

