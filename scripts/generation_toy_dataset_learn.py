from neighbors_and_roc.options import Options, Data_Defaults, Model_Defaults
from neighbors_and_roc import util_ROC, fixed_settings, util_text, util_emb, my_models

import random
import json
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import string
#
# # OPTIONS
# TO_WRITE = os.path.join(fixed_settings.GENERATED_DATA_ROOT, 'generation_yelplike.txt')
# data_opts = Data_Defaults()
# data_opts.NUM_STORIES = 10000
# print(data_opts, "\n")
# opts = Toy_Defaults()
# opts.EMBEDDINGS_FILENAME = 'GoogleNews-vectors-negative300.txt'
# opts.EMBEDDINGS_FILEPATH = os.path.join(fixed_settings.EMBEDDINGS_ROOT, opts.EMBEDDINGS_FILENAME) if opts.EMBEDDINGS_FILENAME != None else None
# opts.HIDDEN_LAYERS = [100]
# # self.BATCH_SIZE = 32
# opts.EPOCHS = 100
# opts.DROPOUT = False
# opts.REGULARIZE = False
# opts.BASE_NUM_TRAINING_SAMPLES = 5000
# opts.PERCENTAGE_TO_ADD = 0
# opts.NUM_TESTING_SAMPLES = 500
# # model_opts.USING_ALTERNATIVES = True
# # model_opts = Model_Defaults() # For quick debugging only
# print(opts, "\n")
# assert(data_opts.NUM_STORIES >
#        opts.BASE_NUM_TRAINING_SAMPLES + opts.PERCENTAGE_TO_ADD * opts.BASE_NUM_TRAINING_SAMPLES + opts.NUM_TESTING_SAMPLES)

# SETTINGS
class Toy_Options(Options):
    def __init__(self):
        self.START_TOKEN = 'start'
        self.STOP_TOKEN = 'stop'

        self.SEQUENCE_LENGTH = 1
        self.VOCAB_SIZE = 10
        self.NUM_TRAINING_SAMPLES = 1000
        self.NUM_TEST_SAMPLES = 1000
        self.EMBEDDING_SIZE = 100

        self.HIDDEN_LAYERS = [20]
        self.DROPOUT = False
        self.REGULARIZE = False
        self.EPOCHS = 2
        self.BATCH_SIZE = 32
opts = Toy_Options()

# TOY DATA
tokens = list(range(opts.VOCAB_SIZE))
in_sentences = [' '.join([opts.START_TOKEN]+[str(random.choice(tokens)) for token_to_gen in range(opts.SEQUENCE_LENGTH)]+[opts.STOP_TOKEN])
                for sentence_to_gen in range(opts.NUM_TRAINING_SAMPLES)]
out_sentences = in_sentences
print("Dataset:")
print('\n'.join(['/'.join(sample) for sample in zip(in_sentences, out_sentences)]))

#region data preparation
# prepare the tokenizer on the source text
in_tokenizer = Tokenizer()
in_tokenizer.fit_on_texts(in_sentences)
in_index_to_word = {index: word for (word, index) in in_tokenizer.word_index.items()}
in_vocab_size = len(in_tokenizer.word_index) + 1  # Because tokenizer does not assign 0
print('Vocabulary Size: %d' % in_vocab_size)

# Convert from text to sequences
in_sequences = in_tokenizer.texts_to_sequences(in_sentences)
in_max_sentence_length = max([len(seq) for seq in in_sequences])
in_sequences = pad_sequences(in_sequences, maxlen=in_max_sentence_length, padding='pre')
print('Max Sequence Length: %d' % in_max_sentence_length)

# prepare the tokenizer on the source text
out_tokenizer = Tokenizer()
out_tokenizer.fit_on_texts(out_sentences)
out_index_to_word = {index: word for (word, index) in out_tokenizer.word_index.items()}
out_vocab_size = len(out_tokenizer.word_index) + 1  # Because tokenizer does not assign 0
print('Vocabulary Size: %d' % out_vocab_size)

# Convert from text to sequences
out_sequences = out_tokenizer.texts_to_sequences(out_sentences)
# pad input sequences
out_max_sentence_length = max([len(seq) for seq in out_sequences])
out_sequences = pad_sequences(out_sequences, maxlen=out_max_sentence_length, padding='post')
print('Max Sequence Length: %d' % out_max_sentence_length)

# Shuffle
data = np.hstack((in_sequences,out_sequences))
# np.random.shuffle(data)
X = data[:,:in_max_sentence_length]
Y = data[:, in_max_sentence_length:]
# print([[in_index_to_word[i] for i in row if i!=0] for row in X]) # HELPFUL FOR DEBUGGING
#endregion

basic_training_data = (X[:opts.NUM_TRAINING_SAMPLES], Y[:opts.NUM_TRAINING_SAMPLES])
testing_data = (X[-1*opts.NUM_TEST_SAMPLES:], Y[-1*opts.NUM_TEST_SAMPLES:])

# TRAIN WITH DATA ADDED
print("Building model...")
# Random embedding, mostly for debugging
training_model, encoder_model, decoder_model = my_models.seq2seq_model(
                    input_length=X.shape[1], output_length=Y.shape[1],
                    in_embedding_matrix_shape=(in_vocab_size, opts.EMBEDDING_SIZE),
                    out_embedding_matrix_shape=(out_vocab_size, opts.EMBEDDING_SIZE),
                    hidden_layers=opts.HIDDEN_LAYERS, dropout=opts.DROPOUT, regularize=opts.REGULARIZE)
training_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model...")
encoder_input_data = basic_training_data[0]
decoder_target_data = basic_training_data[1]

### Tab over 2d array with tab=c ###
def tab(arr, c):
    (rows,cols) = arr.shape
    tabs = np.array([c]*rows).reshape(rows,1)
    return np.hstack((tabs,arr[:,:-1]))

### Convert elements of 2d array so now the 3rd dimension in the one hots ###
def to_cat(arr):
    # (num_samples, len_sequence) = arr.shape
    np.asarray([to_categorical(sample, num_classes=out_vocab_size) for sample in arr])

training_model.fit(
        [encoder_input_data, tab(decoder_target_data, 0)], to_categorical(decoder_target_data),
        epochs=opts.EPOCHS, batch_size=opts.BATCH_SIZE, verbose=2)
print("Evaluating model...")
num_correct = 0
for i in range(opts.NUM_TEST_SAMPLES):
    x,y = testing_data[0][i,:].reshape((1,testing_data[0].shape[1])), testing_data[1][i,:].reshape((1,testing_data[1].shape[1]))
    generated = my_models.decode_sequence(x, Y.shape[1],
                                          encoder_model, decoder_model,
                                          out_tokenizer.word_index, out_index_to_word,
                                          start_token=opts.START_TOKEN, stop_token=opts.STOP_TOKEN)
    print(generated)
#     prediction = model.predict_classes(x)[0]
#     if y[0,prediction] == 1: print('\nCorrect:'); num_correct += 1
#     else: print('\nWrong')
#     choice_num = 0
#     for start in range(0,x.shape[1],max_sentence_length):
#         predicted_flag = ''
#         gold_star_flag = ''
#         if start != 0:
#             if choice_num == prediction: predicted_flag = '>'
#             if y[0,choice_num] == 1: gold_star_flag = '*'
#             choice_num += 1
#         print(gold_star_flag+predicted_flag+
#               ' '.join([index_to_word[i] for i in x[0,start:start+max_sentence_length] if i!=0])) # HELPFUL FOR DEBUGGING
# print('\n{correct}/{all}={perc:.2%}'.format(correct=num_correct, all=model_opts.NUM_TESTING_SAMPLES,
#                                         perc=num_correct/model_opts.NUM_TESTING_SAMPLES))

