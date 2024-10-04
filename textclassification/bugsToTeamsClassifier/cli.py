import tensorflow as tf
import keras_nlp
import json
import numpy as np
from normalization import normalizeSingleText

MAX_SEQUENCE_LENGTH = 40

def load_model(model_path):
    # Load the FNet model from the saved file
    model = tf.keras.models.load_model(model_path, custom_objects={"FNetEncoder": keras_nlp.layers.FNetEncoder})
    return model


def load_tokenizer():
    with open("./vocab.json", 'r') as f:
        vocab = json.load(f)

    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=False,
        sequence_length=MAX_SEQUENCE_LENGTH,
    )
    print(tokenizer("what it is"))
    return tokenizer

def preprocess_text(text, tokenizer, seq_length=128):
    # Lowercase and tokenize the text
    tokenized_input = tokenizer(text)
    print("tokenized_input")
    print(tokenized_input)
    token_ids = tokenized_input

    # Ensure the token_ids are padded or truncated to the desired sequence length
    if len(token_ids) > seq_length:
        token_ids = token_ids[:seq_length]
    else:
        token_ids = np.pad(token_ids, (0, seq_length - len(token_ids)), 'constant', constant_values=0)
    return np.array([token_ids])


def classify_text(model, tokenizer, text):
    # Preprocess the input text
    normalizeSingleText(text)
    processed_input = preprocess_text(text, tokenizer)
    print(processed_input)
    y_prob = model.predict(processed_input)
    y_classes = y_prob.argmax(axis=-1)
    with open('classnames.json', 'r') as file:
        data = json.load(file)
    print(y_classes)
    print(data[y_classes[0]])
    return data[y_classes[0]]

model_path = "model.keras"
config_path = "tokenizer_config.json"
model = load_model(model_path)
# Load the tokenizer configuration and reconstruct the tokenizer
tokenizer = load_tokenizer(config_path)

# Classify the text
with open('input.txt', 'r') as file:
    input_text = file.read()

result = classify_text(model, tokenizer, input_text)
print(f"Classification: {result}")
