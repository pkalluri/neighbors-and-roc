from keras.layers import Input, Embedding, Dense, Activation, Flatten, LSTM,Dropout
from keras.models import Model, Sequential
from keras import regularizers
import numpy as np

def classifier_model(input_length, num_classes, embedding_matrix_shape=None, embedding_matrix=None, hidden_layers=[], dropout=False, regularize=False):
    # Create embedding layer
    if embedding_matrix is None: # Initialize embedding randomly
        (VOCABULARY_SIZE, EMBEDDING_SIZE) = embedding_matrix_shape
        embedding_layer = Embedding(VOCABULARY_SIZE,
                                    EMBEDDING_SIZE,
                                    input_length=input_length,
                                    trainable=True)
    else:
        (VOCABULARY_SIZE, EMBEDDING_SIZE) = embedding_matrix.shape
        embedding_layer = Embedding(VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=True)

    kernel_regularizer = regularizers.l2(.01) if regularize else None
    activity_regularizer = None

    # Create model
    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=input_length))
    if dropout: model.add(Dropout(.5))
    if(len(hidden_layers)==0):
        model.add(LSTM(num_classes, activation='softmax'))
        model.add(LSTM(num_classes, activation='softmax', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer))
    else:
        model.add(LSTM(hidden_layers[0], activation='tanh', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer))
        if dropout: model.add(Dropout(.5))
        for size in hidden_layers[1:]:
            model.add(Dense(size,activation='tanh', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer))
            if dropout: model.add(Dropout(.5))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer))
    model.summary()
    return model


def seq2seq_model(input_length, output_length, embedding_matrix_shape=None, embedding_matrix=None, hidden_layers=[], dropout=False, regularize=False):
    ### Code modified from https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html ###

    #  Create embedding layer
    if embedding_matrix is None: # Initialize embedding randomly
        (VOCABULARY_SIZE, EMBEDDING_SIZE) = embedding_matrix_shape
        embedding_layer = Embedding(VOCABULARY_SIZE,
                                    EMBEDDING_SIZE,
                                    input_length=input_length,
                                    trainable=True)
    else:
        (VOCABULARY_SIZE, EMBEDDING_SIZE) = embedding_matrix.shape
        embedding_layer = Embedding(VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=True)

    kernel_regularizer = regularizers.l2(.01) if regularize else None
    activity_regularizer = regularizers.l1(.01) if regularize else None
    dim = hidden_layers[0] #Todo handle other requested hidden layers

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, ))
    emb = embedding_layer(encoder_inputs)
    _, state_h, state_c = LSTM(dim, return_state=True)(emb)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    emb = embedding_layer(decoder_inputs)
    decoder_lstm_layer = LSTM(dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm_layer(emb, initial_state=encoder_states)
    decoder_dense_layer = Dense(output_length, activation='softmax')
    decoder_outputs = decoder_dense_layer(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(dim,))
    decoder_state_input_c = Input(shape=(dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm_layer(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense_layer(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

    training_model.summary()
    return training_model, encoder_model, decoder_model

def decode_sequence(input_seq, output_length, encoder_model, decoder_model, word_to_index, index_to_word):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, output_length))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word_to_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > output_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, output_length))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence