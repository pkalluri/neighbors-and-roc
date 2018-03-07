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
                                    trainable=False)
    else:
        (VOCABULARY_SIZE, EMBEDDING_SIZE) = embedding_matrix.shape
        embedding_layer = Embedding(VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    # kernel_regularizer = regularizers.l2(.01) if regularize else None
    # activity_regularizer = None

    # Create model
    model = Sequential()
    model.add(embedding_layer)
    # if dropout: model.add(Dropout(.5))
    if(len(hidden_layers)==0):
        model.add(LSTM(num_classes, activation='softmax', dropout=dropout, recurrent_dropout=dropout))
    else:
        model.add(LSTM(hidden_layers[0], activation='tanh'))
        if dropout: model.add(Dropout(.5))
        for size in hidden_layers[1:]:
            model.add(Dense(size,activation='tanh'))
            if dropout: model.add(Dropout(.5))
        model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def seq2seq_model(input_length, output_length,
                  in_embedding_matrix_shape=None, in_embedding_matrix=None,
                  out_embedding_matrix_shape=None, out_embedding_matrix=None,
                  hidden_layers=[], dropout=False, regularize=False):
    ### Code modified from https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html ###

    if in_embedding_matrix is None: # Initialize embedding randomly
        (IN_VOCABULARY_SIZE, EMBEDDING_SIZE) = in_embedding_matrix_shape
        in_embedding_layer = Embedding(IN_VOCABULARY_SIZE,
                                    EMBEDDING_SIZE,
                                    input_length=input_length,
                                    trainable=False)
    else:
        (IN_VOCABULARY_SIZE, EMBEDDING_SIZE) = in_embedding_matrix.shape
        in_embedding_layer = Embedding(IN_VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[in_embedding_matrix],
                                input_length=input_length,
                                trainable=False)
    if out_embedding_matrix is None: # Initialize embedding randomly
        (OUT_VOCABULARY_SIZE, EMBEDDING_SIZE) = out_embedding_matrix_shape
        out_embedding_layer = Embedding(OUT_VOCABULARY_SIZE,
                                    EMBEDDING_SIZE,
                                    input_length=output_length,
                                    trainable=False)
    else:
        (OUT_VOCABULARY_SIZE, EMBEDDING_SIZE) = out_embedding_matrix.shape
        out_embedding_layer = Embedding(OUT_VOCABULARY_SIZE,
                                EMBEDDING_SIZE,
                                weights=[out_embedding_matrix],
                                input_length=output_length,
                                trainable=False)


    kernel_regularizer = regularizers.l2(.01) if regularize else None
    activity_regularizer = regularizers.l1(.01) if regularize else None
    dim = hidden_layers[0] #Todo handle other requested hidden layers

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, ), name="encoder_inputs")
    encoder_emb = in_embedding_layer(encoder_inputs)
    _, state_h, state_c = LSTM(dim, return_state=True,
                               dropout=dropout, recurrent_dropout=dropout, name="encoder_LSTM",
                               kernel_regularizer=kernel_regularizer)(encoder_emb)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, ), name="decoder_inputs")
    decoder_emb = out_embedding_layer(decoder_inputs)
    decoder_lstm_layer = LSTM(dim, return_sequences=True,
                              dropout=dropout, recurrent_dropout=dropout, name="decoder_lstm",
                              kernel_regularizer=kernel_regularizer)
    decoder_outputs = decoder_lstm_layer(decoder_emb, initial_state=encoder_states)
    decoder_dense_layer = Dense(OUT_VOCABULARY_SIZE, activation='softmax', kernel_regularizer=kernel_regularizer)
    decoder_outputs = decoder_dense_layer(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(dim,))
    decoder_state_input_c = Input(shape=(dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm_layer(decoder_emb, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense_layer(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

    training_model.summary()
    return training_model, encoder_model, decoder_model

def decode_sequence(input_seq, output_length, encoder_model, decoder_model, word_to_index, index_to_word, start_token, stop_token):
    output_space_size = len(word_to_index)

    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, output_space_size))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, word_to_index[start_token]] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) # sample, seq, all probs
        sampled_token = index_to_word[sampled_token_index]
        decoded_sentence += sampled_token

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token == stop_token or len(decoded_sentence) > output_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, output_space_size))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence