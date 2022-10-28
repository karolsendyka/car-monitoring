from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from pathlib import Path
import data_ingester

# Fails because path is bad ;/
import pi_eye

app = Flask(__name__)
UPLOAD_DIR = "uploads/"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

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


# import os
@app.route("/load", methods=['GET'])
def load_input_files():
    data_ingester.load_input_files(UPLOAD_DIR)
    loaded_data = data_ingester.list()
    result = ""
    for observation in loaded_data:
        result = result + f"{observation[0]} {str(observation[1])} {str(observation[2])} \n"
    return result
