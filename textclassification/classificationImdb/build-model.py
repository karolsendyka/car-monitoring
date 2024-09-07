import keras_nlp
import keras
import tensorflow as tf
import os
import json

keras.utils.set_random_seed(42)

BATCH_SIZE = 64
EPOCHS = 1
MAX_SEQUENCE_LENGTH = 512
VOCAB_SIZE = 15000

EMBED_DIM = 128
INTERMEDIATE_DIM = 512

print(os.listdir("./aclImdb"))
print(os.listdir("./aclImdb/train"))
print(os.listdir("./aclImdb/test"))

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42,
)
val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42,
)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=BATCH_SIZE)




# to lower case
train_ds = train_ds.map(lambda x, y: (tf.strings.lower(x), y))
val_ds = val_ds.map(lambda x, y: (tf.strings.lower(x), y))
test_ds = test_ds.map(lambda x, y: (tf.strings.lower(x), y))

#Print samples
for text_batch, label_batch in train_ds.take(1):
    for i in range(3):
        print(text_batch.numpy()[i])
        print(label_batch.numpy()[i])



# Tokenize
def train_word_piece(ds, vocab_size, reserved_tokens):
    word_piece_ds = ds.unbatch().map(lambda x, y: x)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        word_piece_ds.batch(1000).prefetch(2),
        vocabulary_size=vocab_size,
        reserved_tokens=reserved_tokens,
    )
    return vocab


reserved_tokens = ["[PAD]", "[UNK]"]
train_sentences = [element[0] for element in train_ds]
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
    mask_zero=True,
)(input_ids)
#
# x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
# x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
# x = keras_nlp.layers.FNetEncoder(intermediate_dim=INTERMEDIATE_DIM)(inputs=x)
#
#
# x = keras.layers.GlobalAveragePooling1D()(x)
# x = keras.layers.Dropout(0.1)(x)
# outputs = keras.layers.Dense(1, activation="sigmoid")(x)
#
# fnet_classifier = keras.Model(input_ids, outputs, name="fnet_classifier")
# # train the model
#
# fnet_classifier.summary()
# fnet_classifier.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     loss="binary_crossentropy",
#     metrics=["accuracy"],
# )
# fnet_classifier.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
#
# fnet_classifier.save("model.keras")


# calculate accuracy
fnet_classifier.evaluate(test_ds, batch_size=BATCH_SIZE)

