from keras.layers import Input, Embedding, Dense, Activation, Flatten, LSTM,Dropout
from keras.models import Model, Sequential
from keras import regularizers

def model(input_length, num_classes, embedding_matrix_shape=None, embedding_matrix=None, hidden_layers=[]):
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
    # Create model
    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=input_length))
    model.add(Dropout(.5))
    if(len(hidden_layers)==0):
        model.add(LSTM(num_classes, activation='softmax'))
                       # kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(.01)))
    else:
        model.add(LSTM(hidden_layers[0], activation='tanh'))
        model.add(Dropout(.5))
        for size in hidden_layers[1:]:
            model.add(Dense(size,activation='tanh'))
            model.add(Dropout(.5))
        model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model