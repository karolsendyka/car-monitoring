import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os.path

TFLITE_FILE_PATH = './pi_eye/model.tflite'
CLASS_NAMES = ['Blekitne suzuki', 'bialy bus', 'chrupek', 'czarne suzuki', 'czerwona mazda', 'srebrny golf']
debug = False

if not os.path.isfile(TFLITE_FILE_PATH):
    raise Exception(f'No file {os.path.abspath(TFLITE_FILE_PATH)}')

print("Loading interpreter ...")
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def classify(img_path):
    if not os.path.isfile(img_path):
        raise Exception(f'No file {os.path.abspath(img_path)}')

    img = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(128, 128)
    )
    showPhoto(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array.astype(np.uint8), 0)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    return CLASS_NAMES[np.argmax(output_data)]


def showPhoto(img):
    if debug:
        plt.imshow(img)
        plt.show()


# returns list of files with classifications
def load_input_files(upload_dir):
    result = {}
    list_of_files_to_process = os.listdir(upload_dir)
    for file in list_of_files_to_process:
        absolute_path = os.path.abspath(upload_dir + "/" + file)
        result[file] = classify(absolute_path)
    return result
