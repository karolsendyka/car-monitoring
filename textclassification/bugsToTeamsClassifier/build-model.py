import json

import keras
import keras_nlp
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from normalization import normalizeText

keras.utils.set_random_seed(17)
BATCH_SIZE = 16
EPOCHS = 50
MAX_SEQUENCE_LENGTH = 40
VOCAB_SIZE = 2000
EMBED_DIM = 128
INTERMEDIATE_DIM = 256

train_ds = keras.utils.text_dataset_from_directory(
    "input/data/train",
    batch_size=BATCH_SIZE,
    validation_split=0.3,
    subset="training",
    seed=17,
)
val_ds = keras.utils.text_dataset_from_directory(
    "input/data/train",
    batch_size=BATCH_SIZE,
    validation_split=0.3,
    subset="validation",
    seed=17,
)
test_ds = keras.utils.text_dataset_from_directory("input/data/validation", batch_size=BATCH_SIZE)
num_classes = len(train_ds.class_names)

with open("classnames.json", "w") as f:
    json.dump(train_ds.class_names, f)

train_ds = normalizeText(train_ds)
val_ds = normalizeText(val_ds)
test_ds = normalizeText(test_ds)

#Print samples
for text_batch, label_batch in train_ds.take(1):
    for i in range(1):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])

# Tokenize
def train_word_piece(ds, vocab_size, reserved_tokens):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(10).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab

reserved_tokens = ["[PAD]", "[UNK]"]
vocab = train_word_piece(train_ds, VOCAB_SIZE, reserved_tokens)
print("Tokens: ", vocab[10:20])

with open("vocab.json", "w") as f:
    f.write(json.dumps(vocab))

#Define tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)

# Save the tokenizer configuration
tokenizer_config = tokenizer.get_config()
with open("tokenizer_config.json", "w") as f:
    f.write(json.dumps(tokenizer_config))

input_sentence_ex = train_ds.take(1).get_single_element()[0][0]
input_tokens_ex = tokenizer(input_sentence_ex)

print("Sentence: ", input_sentence_ex)
print("Tokens: ", input_tokens_ex)
print("Recovered text after detokenizing: ", tokenizer.detokenize(input_tokens_ex))


# formatting the dataset
def format_dataset(sentence, label):
    sentence = tokenizer(sentence)
    return ({"input_ids": sentence}, label)

def make_dataset(dataset):
    dataset = dataset.map(format_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(16).cache()

train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)
test_ds = make_dataset(test_ds)

#Build the model
input_ids = keras.Input(shape=(None,), dtype="int64", name="input_ids")

x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=False,
)(input_ids)

l2_reg = keras.regularizers.L2(0.0003)  # You can tune this value

inputs = keras.Input(shape=(None,), dtype='int32', name='input_ids')

# LSTM layer
x = keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM)(inputs)  # Use appropriate values for vocab_size and embedding_dim
x = keras.layers.LSTM(units=INTERMEDIATE_DIM, return_sequences=True)(x)  # Change INTERMEDIATE_DIM as needed

# Global Average Pooling and Dropout
x = keras.layers.GlobalAveragePooling1D()(x)
# x = keras.layers.Dropout(0.4)(x)

# Output layer
outputs = keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2_reg)(x)

# Create the model
classifier = keras.Model(inputs, outputs, name="lstm_classifier")
classifier.summary()

# Compile the model
classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,  # Number of epochs to wait for improvement
    restore_best_weights=True,
)

# Fit the model

history = classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stopping])
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

classifier.save("model.keras")

# calculate accuracy
classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

def plot_and_save_history(history, filename='training_history.png'):
    # Accuracy plot
    plt.figure(figsize=(12, 5))

    # Subplot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Subplot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Save plot as PNG file
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()

plot_and_save_history(history.history, 'training_history.png')

# COMMITED
# 42/42 ━━━━━━━━━━━━━━━━━━━━ 2s 37ms/step - accuracy: 0.8670 - loss: 0.5292 - val_accuracy: 0.5461 - val_loss: 2.4518
# 3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 10ms/step - accuracy: 0.0743 - loss: 5.7053


# LAST



# NEXT
# EXPERIMENT WITH BATCH SIZES
# DATA AUGUMENTATION
# Tune the number of units in the LSTM layer or consider adding more LSTM layers (with smaller units) for deeper feature extraction.
