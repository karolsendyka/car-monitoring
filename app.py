from flask import Flask

app = Flask(__name__)


@app.route("/event", methods = ['POST'])
def register_event():
    app.logger.info('A value for debugging1')
    return { "hello": "world1"}


@app.route("/event", methods = ['GET'])
def get_event():
    app.logger.info('A value for debugging2')
    return { "hello": "world2"}
