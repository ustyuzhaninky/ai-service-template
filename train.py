#!/usr/bin/python3
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__file__)
import tensorflow as tf; print("Tensorflow", tf.__version__)
import os
import time

from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_hub as hub
import tensorflow_text as text

tf.get_logger().setLevel('ERROR')

# Read, then decode for py2 compat.
path_to_file = tf.keras.utils.get_file('shakespeare.txt',
'http://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# The unique characters in the file
vocab = sorted(set(text))

## Some Variables
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000
TEST_SIZE = 1000

# Length of the vocabulary in chars
VOCAB_SIZE = len(vocab)

# The embedding dimension
EMBEDDING_DIM = 256

# Number of RNN units
RNN_UNITS = 1024

# Creating an extractor for IDs from vocab
ids_from_chars = preprocessing.StringLookup(
        vocabulary=list(vocab), mask_token=None)

# Creating convertor of IDs into characters
chars_from_ids = preprocessing.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True, mask_token=None)

class GenModelv0(tf.keras.Model):
    def __init__(self,
                vocab_size,
                embedding_dim,
                rnn_units):
        super().__init__(self)
        
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
            return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units

    def call(self,
            inputs,
            states=None,
            return_state=False,
            training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(
            x,
            initial_state=states,
            training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
    
    def get_config(self):
        base_config = super().get_config()
        base_config["embedding"] = self.embedding.get_config()
        base_config["gru"] = self.gru.get_config()
        base_config["dense"] = self.dense.get_config()
        base_config["vocab_size"] = self.vocab_size
        base_config["embedding_dim"] = self.embedding_dim
        base_config["rnn_units"] = self.rnn_units
        return base_config
    
    @classmethod
    def from_config(cls, config):
        cls = cls(**config)
        cls.embedding = tf.keras.layers.Embedding(
            config["vocab_size"],
            config["embedding_dim"])
        cls.gru = tf.keras.layers.GRU(
            config["rnn_units"],
            return_sequences=True,
            return_state=True)
        cls.dense = tf.keras.layers.Dense(config["vocab_size"])
        cls.vocab_size = config["vocab_size"]
        cls.embedding_dim = config["embedding_dim"]
        cls.rnn_units = config["rnn_units"]

        return cls

def prepare_training_dataset(text:str):
    converted_string = ids_from_chars(
        tf.strings.unicode_split([text], 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(
        converted_string[0])
    seq_length = 100
    sequences = ids_dataset.batch(
        seq_length+1, drop_remainder=True)
    ids_dataset = tf.data.Dataset.from_tensor_slices(tf.strings.unicode_split([text], 'UTF-8')[0])

    sequences = ids_dataset.batch(
        BATCH_SIZE, drop_remainder=True)
    
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    dataset = sequences.map(split_input_target)
    return dataset

class OneStep(tf.keras.Model):
    def __init__(self,
                 model,
                 chars_from_ids,
                 ids_from_chars,
                 temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(
            ['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(
                ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(
            sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        inputs = str(inputs)
        input_chars = tf.strings.unicode_split(
            inputs, 'UTF-8')
        
        input_ids = self.ids_from_chars(
            input_chars)
        input_ids = tf.expand_dims(input_ids, 0)

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states,
            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(
            predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(
            predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = tf.strings.reduce_join(
            self.chars_from_ids(
            predicted_ids), axis=-1)

        # Return the characters and model state.
        return predicted_chars, states

def generate_text(model,
                  n_characters=1000,
                  query='ROMEO:',
                  chars_from_ids=chars_from_ids,
                  ids_from_chars=ids_from_chars):
    start = time.time()
    states = None
    next_char = query
    result = [next_char]
    
    one_step_model = OneStep(model,
        chars_from_ids, ids_from_chars)

    for n in range(n_characters):
        next_char = tf.strings.join(result)
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()

    return {'text':result[0].numpy().decode('utf-8'),
            'runtime': end - start}

def get_model():
    tfhub_handle_encoder = "small_bert_en_uncased"
    tfhub_handle_preprocess = "bert_en_uncased_preprocess_3"
    text_input = tf.keras.layers.Input(shape=(100,), dtype=tf.int32, name='text')
    # preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    # preprocessing_layer = tf.keras.layers.Embedding(
            # VOCAB_SIZE, EMBEDDING_DIM, name='preprocessing')
    encoder_inputs = text_input#preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
    model = tf.keras.Model(text_input, net)
    # model = GenModelv0(
    #     # Be sure the vocabulary size matches the `StringLookup` layers.
    #     vocab_size=len(ids_from_chars.get_vocabulary()),
    #     embedding_dim=EMBEDDING_DIM,
    #     rnn_units=RNN_UNITS)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [
        tf.metrics.sparse_categorical_accuracy
    ]
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=metrics)
    return model

def train(epochs=1):

    train_text = text[:int(len(text)*0.8)]
    with open("train.txt", 'wb') as train:
        train.write(train_text.encode(encoding='utf-8'))

    test_text = text[-int(len(text)*0.2):]
    with open("test.txt", 'wb') as test:
        test.write(test_text.encode(encoding='utf-8'))

    dataset = prepare_training_dataset(train_text)
    test_dataset = prepare_training_dataset(test_text)
    dataset = (
        dataset
        # .shuffle(BUFFER_SIZE)
        # .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    test_dataset = (
        test_dataset
        # .shuffle(BUFFER_SIZE)
        # .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))
    
    model = get_model()

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard()
    history = model.fit(
        dataset,
        epochs=epochs,
        callbacks=[checkpoint_callback, tensorboard_callback],
        validation_data=test_dataset)
    model.save_weights(os.path.join(os.environ["MODEL_DIR"],
                os.environ["MODEL_FILE"]))
    
    one_step_model = OneStep(model, 
        chars_from_ids, ids_from_chars)

    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]

    for n in range(100):
        next_char, states = one_step_model.generate_one_step(tf.strings.join(result), states=states)
        result.append(next_char)
    print(tf.strings.join(result)[0].numpy().decode("utf-8"))

if __name__ == '__main__':
    os.environ["MODEL_DIR"] = ''
    os.environ["MODEL_FILE"] = 'model.tf'
    if len(sys.argv) > 1:
        epochs = sys.argv[1] if len(sys.argv) >= 1 else 20
        if not isinstance(epochs, int):
            epochs = 20
    else:
        epochs = 20
    train(epochs)
