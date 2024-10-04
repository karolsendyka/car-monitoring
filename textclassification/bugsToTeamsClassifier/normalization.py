import tensorflow as tf

def normalizeText(ds):
    return ds.map(lambda x, y: (normalizeSingleText(x), y))

def normalizeSingleText(x):
    return tf.strings.lower(x)