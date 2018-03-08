from neighbors_and_roc.options import Data_Defaults, Model_Defaults
from neighbors_and_roc import util_ROC, fixed_settings, util_text, util_emb, my_models

import random
import json
import os
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np

# OPTIONS
TO_WRITE = os.path.join(fixed_settings.GENERATED_DATA_ROOT, 'generation_yelplike.txt')
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 10000
print(data_opts, "\n")
model_opts = Model_Defaults()
model_opts.EMBEDDINGS_FILENAME = 'GoogleNews-vectors-negative300.txt'
model_opts.EMBEDDINGS_FILEPATH = os.path.join(fixed_settings.EMBEDDINGS_ROOT, model_opts.EMBEDDINGS_FILENAME) if model_opts.EMBEDDINGS_FILENAME != None else None
model_opts.HIDDEN_LAYERS = [100]
# self.BATCH_SIZE = 32
model_opts.EPOCHS = 100
model_opts.DROPOUT = False
model_opts.REGULARIZE = False
model_opts.BASE_NUM_TRAINING_SAMPLES = 5000
model_opts.PERCENTAGE_TO_ADD = 0
model_opts.NUM_TESTING_SAMPLES = 500
# model_opts.USING_ALTERNATIVES = True
# model_opts = Model_Defaults() # For quick debugging only
print(model_opts, "\n")
assert(data_opts.NUM_STORIES >
       model_opts.BASE_NUM_TRAINING_SAMPLES + model_opts.PERCENTAGE_TO_ADD*model_opts.BASE_NUM_TRAINING_SAMPLES + model_opts.NUM_TESTING_SAMPLES)


# DATA
with open(os.path.join(fixed_settings.DATA_ROOT,'yelp-10000.json'), 'r') as file:
    reviews = [json.loads(line) for line in file]
stars_to_simple_responses = {}
stars_to_simple_responses[5]=['I am impressed!', 'I will be back.']
stars_to_simple_responses[4]=['I am impressed!', 'I will be back.']
stars_to_simple_responses[2]=['Not impressed.', 'Will not be back.']
stars_to_simple_responses[1]=['Not impressed.', 'Will not be back.']
X_sentences = []
Y_sentences = []
Y_alternative_sentences = []
for review in reviews:
    curr_sentences = util_text.get_sentences(review['text'])
    sentence = curr_sentences[min(1, len(curr_sentences) - 1)]
    sentence = ' '.join(sentence.split())
    stars = review['stars']
    if sentence is not '' and stars in stars_to_simple_responses:
        X_sentences.append(sentence)
        r = random.randint(0,1)
        Y_sentences.append(stars_to_simple_responses[stars][r])
        Y_alternative_sentences.append(stars_to_simple_responses[stars][1-r])
print('\n\n'.join(['\n'.join(t) for t in zip(X_sentences, Y_sentences, Y_alternative_sentences)]), file=open(TO_WRITE, 'w'))
#region data preparation
# prepare the tokenizer on the source text #TODO see what can be moved out of this file
in_tokenizer = Tokenizer()
in_tokenizer.fit_on_texts(X_sentences)
in_index_to_word = {index: word for (word, index) in in_tokenizer.word_index.items()}
# determine the vocabulary size
in_vocab_size = len(in_tokenizer.word_index) + 1  # Because tokenizer does not assign 0
print('Vocabulary Size: %d' % in_vocab_size)
# Convert from text to sequences
in_sequences = in_tokenizer.texts_to_sequences(X_sentences)
# pad input sequences
in_max_sentence_length = max([len(seq) for seq in in_sequences])
in_sequences = pad_sequences(in_sequences, maxlen=in_max_sentence_length, padding='pre')
print('Max Sequence Length: %d' % in_max_sentence_length)
# split into input and output elements

# prepare the tokenizer on the source text #TODO see what can be moved out of this file
out_tokenizer = Tokenizer()
out_tokenizer.fit_on_texts(Y_sentences+Y_alternative_sentences)
out_index_to_word = {index: word for (word, index) in out_tokenizer.word_index.items()}
# determine the vocabulary size
out_vocab_size = len(out_tokenizer.word_index) + 1  # Because tokenizer does not assign 0
print('Vocabulary Size: %d' % out_vocab_size)
# Convert from text to sequences
Y_sequences = out_tokenizer.texts_to_sequences(Y_sentences)
Y_alternative_sequences = out_tokenizer.texts_to_sequences(Y_alternative_sentences)
# pad input sequences
out_max_sentence_length = max([len(seq) for seq in Y_sequences]+[len(seq) for seq in Y_alternative_sequences])
Y_sequences = pad_sequences(Y_sequences, maxlen=out_max_sentence_length, padding='post')
Y_alternative_sequences = pad_sequences(Y_alternative_sequences, maxlen=out_max_sentence_length, padding='post')
print('Max Sequence Length: %d' % out_max_sentence_length)


data = np.hstack((in_sequences[:, :], Y_sequences[:, :], Y_alternative_sequences[:,:]))
# x y1 y2
# x y1 y2
# ...
# np.random.shuffle(data)

X = data[:,:-2*out_max_sentence_length]
Y = data[:,-2*out_max_sentence_length:-1*out_max_sentence_length]
Y_alternatives = data[:,-1*out_max_sentence_length:]
print(X.shape)
# print([[in_index_to_word[i] for i in row if i!=0] for row in X]) # HELPFUL FOR DEBUGGING
#endregion
basic_training_data = (X[:model_opts.BASE_NUM_TRAINING_SAMPLES],
                       Y[:model_opts.BASE_NUM_TRAINING_SAMPLES])
NUM_SAMPLES_TO_ADD = int(model_opts.BASE_NUM_TRAINING_SAMPLES * model_opts.PERCENTAGE_TO_ADD)
fresh_training_data = (X[model_opts.BASE_NUM_TRAINING_SAMPLES:model_opts.BASE_NUM_TRAINING_SAMPLES + NUM_SAMPLES_TO_ADD],
                       Y[model_opts.BASE_NUM_TRAINING_SAMPLES:model_opts.BASE_NUM_TRAINING_SAMPLES + NUM_SAMPLES_TO_ADD])
alternative_training_data = (X[:NUM_SAMPLES_TO_ADD],
                             Y_alternatives[:NUM_SAMPLES_TO_ADD])
testing_data = (X[len(X) - model_opts.NUM_TESTING_SAMPLES:],
                Y[len(X) - model_opts.NUM_TESTING_SAMPLES:] + Y_alternatives[len(X) - model_opts.NUM_TESTING_SAMPLES:])


# GET EMBEDDINGS
if model_opts.EMBEDDINGS_FILENAME:
    print("Retrieving embeddings...")
    in_embedding_matrix = util_emb.get_embedding_matrix(
        embedding_size=model_opts.EMBEDDING_SIZE, embedding_path=model_opts.EMBEDDINGS_FILEPATH, vocab_size=in_vocab_size, word_to_index=in_tokenizer.word_index)
    out_embedding_matrix = util_emb.get_embedding_matrix(
        embedding_size=model_opts.EMBEDDING_SIZE, embedding_path=model_opts.EMBEDDINGS_FILEPATH, vocab_size=out_vocab_size, word_to_index=out_tokenizer.word_index)

# TRAIN WITH DATA ADDED
additional_training_data = alternative_training_data if model_opts.USING_ALTERNATIVES else fresh_training_data
print("Building model...")
if model_opts.EMBEDDINGS_FILENAME:
    training_model,encoder_model,decoder_model = my_models.seq2seq_model_with_embs(input_length=X.shape[1], output_length=Y.shape[1],
                                                                                   in_embedding_matrix=in_embedding_matrix, out_embedding_matrix=out_embedding_matrix,
                                                                                   hidden_layers=model_opts.HIDDEN_LAYERS, dropout=model_opts.DROPOUT, regularize=model_opts.REGULARIZE)
else:  # Random embedding, mostly for debugging
    training_model, encoder_model, decoder_model = my_models.seq2seq_model_with_embs(input_length=X.shape[1], output_length=Y.shape[1],
                                                                                     in_embedding_matrix_shape=(in_vocab_size, model_opts.EMBEDDING_SIZE),
                                                                                     out_embedding_matrix_shape=(out_vocab_size, model_opts.EMBEDDING_SIZE),
                                                                                     hidden_layers=model_opts.HIDDEN_LAYERS, dropout=model_opts.DROPOUT, regularize=model_opts.REGULARIZE)
training_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model...")
encoder_input_data = np.vstack((basic_training_data[0], additional_training_data[0]))
decoder_target_data = np.vstack((basic_training_data[1], additional_training_data[1]))
tab = np.asarray([0]*decoder_target_data.shape[0]).reshape(decoder_target_data.shape[0],1)
decoder_input_data = np.hstack((tab, decoder_target_data[:,:-1])) # Target data slid over
training_model.fit([encoder_input_data, decoder_input_data], to_categorical(decoder_target_data),
                   epochs=model_opts.EPOCHS, batch_size=model_opts.BATCH_SIZE, verbose=2)
print("Evaluating model...")
# Note that model.evaluate() is intentionally not used here because it will only mark 1 (the first) of the 2 valid answers correct
num_correct = 0
for i in range(model_opts.NUM_TESTING_SAMPLES):
    x,y = testing_data[0][i,:].reshape((1,testing_data[0].shape[1])), testing_data[1][i,:].reshape((1,testing_data[1].shape[1]))
    generated = my_models.decode_sequence(x, Y.shape[1], encoder_model, decoder_model, out_tokenizer.word_index, out_index_to_word)
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

