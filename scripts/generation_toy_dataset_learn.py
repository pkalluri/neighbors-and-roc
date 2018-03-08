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
        # self.EMBEDDING_SIZE = 100

        self.HIDDEN_DIM = 20
        self.DROPOUT = .2
        self.REGULARIZE = False
        self.EPOCHS = 100
        self.BATCH_SIZE = 32
opts = Toy_Options()

for SEQUENCE_LENGTH in [1]:
    opts.SEQUENCE_LENGTH = SEQUENCE_LENGTH
    for VOCAB_SIZE in [10]:
        opts.VOCAB_SIZE = VOCAB_SIZE
        for NUM_TRAINING_SAMPLES in [VOCAB_SIZE*10**(p) for p in [0,1,2]]:
            opts.NUM_TRAINING_SAMPLES = NUM_TRAINING_SAMPLES
            for HIDDEN_DIM in [10,150,300]:
                opts.HIDDEN_DIM = HIDDEN_DIM
                for DROPOUT in [0,.2,.5]:
                    opts.DROPOUT = DROPOUT
                    for BATCH_SIZE in [16,32,64,128]:
                        opts.BATCH_SIZE = BATCH_SIZE
                        #region generate data
                        # TOY DATA
                        tokens = list(range(opts.VOCAB_SIZE))
                        in_sentences = [' '.join([str(random.choice(tokens)) for token_to_gen in range(opts.SEQUENCE_LENGTH)]+[opts.STOP_TOKEN])
                                        for sentence_to_gen in range(opts.NUM_TRAINING_SAMPLES)]
                        out_sentences = in_sentences
                        print("Dataset:")
                        # print('\n'.join(['/'.join(sample) for sample in zip(in_sentences, out_sentences)]))

                        # prepare the tokenizer on the source text
                        in_tokenizer = Tokenizer()
                        in_tokenizer.fit_on_texts(in_sentences+[opts.START_TOKEN])
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
                        out_tokenizer.fit_on_texts(out_sentences+[opts.START_TOKEN])
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
                        Y = data[:,in_max_sentence_length:]
                        # print([[in_index_to_word[i] for i in row if i!=0] for row in X]) # HELPFUL FOR DEBUGGING
                        #endregion
                        training_data = (X[:opts.NUM_TRAINING_SAMPLES], Y[:opts.NUM_TRAINING_SAMPLES])
                        testing_data = (X[-1*opts.NUM_TEST_SAMPLES:], Y[-1*opts.NUM_TEST_SAMPLES:])

                        #region prep data for model
                        # TRAIN WITH DATA ADDED
                        print("Training model...")
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
                        # training_model, encoder_model, decoder_model = my_models.seq2seq_models(
                        #     input_length=X.shape[1], output_length=Y.shape[1],
                        #     in_embedding_matrix_shape=(in_vocab_size, opts.EMBEDDING_SIZE),
                        #     out_embedding_matrix_shape=(out_vocab_size, opts.EMBEDDING_SIZE),
                        #     hidden_layers=opts.HIDDEN_LAYERS, dropout=opts.DROPOUT, regularize=opts.REGULARIZE)
                        training_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        #endregion

                        #region train
                        training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                validation_data=([test_input_data, np.random.normal(test_decoder_input_data)],test_target_data),
                                epochs=opts.EPOCHS, batch_size=opts.BATCH_SIZE, verbose=2)
                        #endregion

                        #region test
                        print("Evaluating model...")
                        # decode a one hot encoded string
                        def one_hot_decode(encoded_seq):
                            return [np.argmax(vector) for vector in encoded_seq]
                        num_correct = 0
                        for i in range(opts.NUM_TEST_SAMPLES):
                            x,y = test_input_data[i,:,:].reshape(1,in_max_sentence_length, in_vocab_size), \
                                  test_target_data[i,:,:].reshape(1, out_max_sentence_length, out_vocab_size)
                            prediction = my_models.predict_sequence(encoder_model, decoder_model, x, out_vocab_size,
                                                                    words_to_index=out_tokenizer.word_index,
                                                                    start_token=opts.START_TOKEN, stop_token=opts.STOP_TOKEN)
                            x_sentence = [out_index_to_word[id] for id in one_hot_decode(x[0])]
                            predicted_sentence = [out_index_to_word[id] for id in one_hot_decode(prediction)]
                            print(x_sentence)
                            print(predicted_sentence)
                            print('\n')
                            if x_sentence == predicted_sentence:
                                num_correct += 1
                        print('{corr}/{all}={perc:.2%}'.format(corr=num_correct, all=opts.NUM_TEST_SAMPLES, perc=num_correct/opts.NUM_TEST_SAMPLES))
                            #
                            # generated = my_models.decode_sequence(x, Y.shape[1],
                            #                                       encoder_model, decoder_model,
                            #                                       out_tokenizer.word_index, out_index_to_word,
                            #                                       start_token=opts.START_TOKEN, stop_token=opts.STOP_TOKEN)
                            # print(generated)
                        #
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

                        #endregion

                        #region save info

                        #endregion