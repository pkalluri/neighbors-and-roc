from neighbors_and_roc.options import Options, Data_Defaults, Model_Defaults
from neighbors_and_roc import util_ROC, fixed_settings, util_text, util_emb, my_models, util_misc, toy_data

import random
import json
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import string
import csv

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

# SETUP
util_misc.force_tf_to_take_memory_only_as_needed() # For use on NLP cluster

#region settings
class Toy_Options(Options):
    def __init__(self):
        self.FUNCTION_TO_MODEL = 'identity'
        self.SEQUENCE_LENGTH = None
        self.VOCAB_SIZE = None
        self.NUM_BASE_TRAINING_SAMPLES = None
        self.PERCENTAGE_SAMPLES_TO_ADD = None
        self.ADDITIONAL_DATA_IS_ALTERNATIVE_DATA = None

        self.NUM_SYNONYMS_PER_TOKEN = 100
        self.NUM_TEST_SAMPLES = 1000

        self.HIDDEN_DIM = 150
        self.DROPOUT = .5
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
START_TOKEN = 'start'
STOP_TOKEN = 'stop'
TO_WRITE = os.path.join(fixed_settings.GENERATED_DATA_ROOT,'runs-comparison-sparse.csv')
new_file = False
on_nlp_cluster = False

opts_list = []
for SEQUENCE_LENGTH in [5]:
    for VOCAB_SIZE in [5]:
        for NUM_BASE_TRAINING_SAMPLES in [1000]:
            for PERCENTAGE_SAMPLES_TO_ADD in [1,3,7,9]:
                for ADDITIONAL_DATA_IS_ALTERNATIVE_DATA in [False,True]:
                    curr_opts = Toy_Options()
                    curr_opts.SEQUENCE_LENGTH = SEQUENCE_LENGTH
                    curr_opts.VOCAB_SIZE = VOCAB_SIZE
                    curr_opts.NUM_BASE_TRAINING_SAMPLES = NUM_BASE_TRAINING_SAMPLES
                    curr_opts.ADDITIONAL_DATA_IS_ALTERNATIVE_DATA = ADDITIONAL_DATA_IS_ALTERNATIVE_DATA
                    curr_opts.PERCENTAGE_SAMPLES_TO_ADD = PERCENTAGE_SAMPLES_TO_ADD
                    opts_list.append(curr_opts)
#endregion

#region write headers
print("Writing to {}".format(TO_WRITE))
rows = []
row = list(vars(Toy_Options()).keys()) + ['Training Accuracy', 'Test Accuracy']
if new_file: # Add header
    if not on_nlp_cluster:
        writing_mode = 'w' if new_file else 'a'
        with open(TO_WRITE, writing_mode) as csvfile:
            writer = csv.writer(csvfile, dialect='excel', delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([str(elt) for elt in row]) # Column headers
    else: rows.append(row)
#endregion

for opts in opts_list:
    print(opts)
    # Only do this run if there's more unique samples than requested training samples
    if ((opts.VOCAB_SIZE*(1+opts.NUM_SYNONYMS_PER_TOKEN))**opts.SEQUENCE_LENGTH) > \
           opts.NUM_BASE_TRAINING_SAMPLES\
           +int(opts.PERCENTAGE_SAMPLES_TO_ADD*opts.NUM_BASE_TRAINING_SAMPLES)\
           +opts.NUM_TEST_SAMPLES:

        #region generate data
        # TOY DATA
        functions = {'identity':toy_data.identity_data}
        in_sentences_train, out_sentences_train = toy_data.identity_data(opts.VOCAB_SIZE, opts.SEQUENCE_LENGTH, opts.NUM_BASE_TRAINING_SAMPLES,
                                                                         opts.PERCENTAGE_SAMPLES_TO_ADD, STOP_TOKEN, opts.NUM_SYNONYMS_PER_TOKEN,
                                                                         opts.ADDITIONAL_DATA_IS_ALTERNATIVE_DATA)
        in_sentences_test, out_sentences_test = toy_data.identity_data(opts.VOCAB_SIZE, opts.SEQUENCE_LENGTH, opts.NUM_TEST_SAMPLES,
                                                                         0, STOP_TOKEN, opts.NUM_SYNONYMS_PER_TOKEN,
                                                                         False)
        print('Dataset: ' + str(list(zip(in_sentences_train, out_sentences_train))[:5]) + '...')

        # prepare the tokenizer on the source text
        in_tokenizer = Tokenizer(filters='')
        in_tokenizer.fit_on_texts(in_sentences_train + [START_TOKEN] + in_sentences_test)
        in_index_to_word = {index: word for (word, index) in in_tokenizer.word_index.items()}
        in_vocab_size = len(in_tokenizer.word_index) + 1  # Because tokenizer does not assign 0
        print('Vocabulary Size: %d' % in_vocab_size)

        # Convert from text to sequences
        in_sequences = in_tokenizer.texts_to_sequences(in_sentences_train + in_sentences_test)
        in_max_sentence_length = max([len(seq) for seq in in_sequences])
        in_sequences = pad_sequences(in_sequences, maxlen=in_max_sentence_length, padding='pre')
        print('Max Sequence Length: %d' % in_max_sentence_length)

        # prepare the tokenizer on the source text
        out_tokenizer = Tokenizer(filters='')
        out_tokenizer.fit_on_texts(out_sentences_train + [START_TOKEN] + out_sentences_test)
        out_index_to_word = {index: word for (word, index) in out_tokenizer.word_index.items()}
        out_vocab_size = len(out_tokenizer.word_index) + 1  # Because tokenizer does not assign 0
        print('Vocabulary Size: %d' % out_vocab_size)

        # Convert from text to sequences
        out_sequences = out_tokenizer.texts_to_sequences(out_sentences_train + out_sentences_test)
        # pad input sequences
        out_max_sentence_length = max([len(seq) for seq in out_sequences])
        out_sequences = pad_sequences(out_sequences, maxlen=out_max_sentence_length, padding='post')
        print('Max Sequence Length: %d' % out_max_sentence_length)

        # Shuffle
        data = np.hstack((in_sequences,out_sequences))
        # np.random.shuffle(data)
        X = data[:,:in_max_sentence_length]
        Y = data[:,in_max_sentence_length:]
        # print([[in_index_to_word[i] for i in row if i!=0] for row in X]) # HELPFUL FOR DEBUGGING
        #endregion
        num_training_samples = opts.NUM_BASE_TRAINING_SAMPLES + int(opts.PERCENTAGE_SAMPLES_TO_ADD*opts.NUM_BASE_TRAINING_SAMPLES)
        training_data = (X[:num_training_samples], Y[:num_training_samples])
        testing_data = (X[num_training_samples:num_training_samples+opts.NUM_TEST_SAMPLES], Y[num_training_samples:num_training_samples+opts.NUM_TEST_SAMPLES])

        #region prep data for model
        # TRAIN WITH DATA ADDED
        ### Tab over 2d array with tab=c ###
        def tab(arr, c):
            (rows,cols) = arr.shape
            tabs = np.array([c]*rows).reshape(rows,1)
            return np.hstack((tabs,arr[:,:-1]))
        ### Convert elements of 2d array so now the 3rd dimension in the one hots ###
        def to_cat(arr, nclasses):
            # (num_samples, len_sequence) = arr.shape
            a = np.asarray([to_categorical(sample, num_classes=nclasses) for sample in arr])
            return a
        #endregion
        encoder_input_data = to_cat(training_data[0], nclasses=in_vocab_size)
        decoder_target_data = to_cat(training_data[1], nclasses=out_vocab_size)
        decoder_input_data = to_cat(tab(training_data[1], 0), nclasses=out_vocab_size)

        test_input_data = to_cat(testing_data[0], nclasses=in_vocab_size)
        test_target_data = to_cat(testing_data[1], nclasses=out_vocab_size)
        test_decoder_input_data = to_cat(tab(testing_data[1], 0), nclasses=out_vocab_size)

        #region define model
        print("Building model...")
        training_model, encoder_model, decoder_model = my_models.seq2seq_models(in_vocab_size, out_vocab_size,
                                                                                hidden_dim=opts.HIDDEN_DIM,
                                                                                dropout=opts.DROPOUT)
        training_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #endregion

        #region train
        print("Training model...")
        history = training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                validation_data=([test_input_data, test_decoder_input_data],test_target_data),
                epochs=opts.EPOCHS, batch_size=opts.BATCH_SIZE, verbose=2)
        #endregion

        #region test
        print("Evaluating model...")
        # decode a one hot encoded string
        def one_hot_decode(encoded_seq):
            return [np.argmax(vector) for vector in encoded_seq]
        num_correct = 0
        for i in range(opts.NUM_TEST_SAMPLES):
            x, target = test_input_data[i, :, :].reshape(1, in_max_sentence_length, in_vocab_size), \
                  test_target_data[i,:,:].reshape(1, out_max_sentence_length, out_vocab_size)
            prediction = my_models.predict_sequence(encoder_model, decoder_model, x, out_vocab_size,
                                                    words_to_index=out_tokenizer.word_index,
                                                    start_token=START_TOKEN, stop_token=STOP_TOKEN,
                                                    prediction_length_cap=out_max_sentence_length)
            x_tokens = [in_index_to_word[id] for id in one_hot_decode(x[0]) if not id==0]
            target_tokens = [out_index_to_word[id] for id in one_hot_decode(target[0]) if not id == 0]
            predicted_tokens = [out_index_to_word[id] for id in one_hot_decode(prediction) if not id == 0]
            if i<3:
                print('x: '+str(x_tokens))
                print('target: '+str(target_tokens))
                print('pred: '+str(predicted_tokens))
            if toy_data.is_paraphrase(target_tokens, predicted_tokens):
                num_correct += 1
                if i<3: print('Correct')
            if i<3: print()
        perc = num_correct / opts.NUM_TEST_SAMPLES
        print('{corr}/{all}={perc:.2%}'.format(corr=num_correct, all=opts.NUM_TEST_SAMPLES, perc=perc))
        #endregion

        #region save info
        history_log = history.history
        row = list(vars(opts).values()) + [history_log["acc"][-1]] + [perc]
        if not on_nlp_cluster:
            with open(TO_WRITE, 'a') as csvfile:
                writer = csv.writer(csvfile, dialect='excel', delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([str(elt) for elt in row])
        else: rows.append(row)
        #endregion

#region writerows at end
if on_nlp_cluster:
    with open(TO_WRITE, writing_mode) as csvfile:
        writer = csv.writer(csvfile, dialect='excel', delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow([str(elt) for elt in row])
#endregion