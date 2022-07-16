from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

# Fails because path is bad ;/
import pi_eye
app = Flask(__name__)
UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR)

@app.route('/upload')
def display_upload_page():
    return render_template('sendimage.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(UPLOAD_DIR + secure_filename(f.filename))

        # returns SpooledTemporaryFile
        print("File?" + str(type(f.stream)))

        # pi_eye.classify(f.stream)

        return 'file uploaded successfully'


@app.route("/events", methods=['POST'])
def register_event():
    app.logger.info('A value for debugging1')
    return {"hello": "world1"}


@app.route("/events", methods=['GET'])
def get_event():
    app.logger.info('A value for debugging2')
    return {"hello": "world2"}
