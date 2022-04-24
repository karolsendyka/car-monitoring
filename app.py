from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/upload')
def display_upload_page():
    return render_template('sendimage.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


@app.route("/events", methods=['POST'])
def register_event():
    app.logger.info('A value for debugging1')
    return {"hello": "world1"}


@app.route("/events", methods=['GET'])
def get_event():
    app.logger.info('A value for debugging2')
    return {"hello": "world2"}
