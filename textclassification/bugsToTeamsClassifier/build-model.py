import json

import keras
import keras_nlp
import tensorflow as tf

from normalization import normalizeText
# tokenizer test
# tf.Tensor([147 146 127 ...   0   0   0], shape=(1024,), dtype=int32)
keras.utils.set_random_seed(17)

BATCH_SIZE = 32
EPOCHS = 50
MAX_SEQUENCE_LENGTH = 256
VOCAB_SIZE = 10000

EMBED_DIM = 128
INTERMEDIATE_DIM = 1024

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

# to lower case
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

#store vocab
with open("vocab.json", "w") as f:
    f.write(json.dumps(vocab))

print("Tokens: ", vocab[10:20])

#Define tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=MAX_SEQUENCE_LENGTH,
)
print("tokenizer test")
print(tokenizer("what it is"))

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

l2_reg = keras.regularizers.L2(0.03)  # You can tune this value

x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
# x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
# x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)


x = keras.layers.GlobalAveragePooling1D()(x)
x = keras.layers.Dropout(0.4)(x)
outputs =keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2_reg)(x)

fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")
fnet_classifier.summary()
fnet_classifier.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,  # Number of epochs to wait for improvement
    restore_best_weights=True,
)
fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stopping])
fnet_classifier.save("model.keras")

# calculate accuracy
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)
