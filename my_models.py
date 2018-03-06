from keras.layers import Input, Embedding, Dense, Activation, Flatten, LSTM
from keras.models import Model, Sequential

def model(input_length, num_classes, embedding_matrix_shape=None, embedding_matrix=None, hidden_layers=[]):
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
    # Create model
    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=input_length))
    model.add(LSTM(hidden_layers[0], activation='relu'))
    for size in hidden_layers[1:]:
        model.add(Dense(size,activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    print(model.summary())
    return model

    # input = Input(shape=(VOCABULARY_SIZE,), dtype='int32')
    # embedding = embedding_layer(input)
    # x = Flatten()(embedding)
    # x = Activation('relu')(x)
    # last_layer_width = EMBEDDING_SIZE
    # for size in hidden_layers:  # Optional additional layers
    #     x = Dense(size, activation='relu')(x)
    #     last_layer_width = last_layer_width / 2
    # preds = Dense(VOCABULARY_SIZE, activation='softmax')(x)
    # model = Model(input, preds)
    # return model