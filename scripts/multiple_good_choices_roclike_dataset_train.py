# File adapted from https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import random

from neighbors_and_roc import util_ROC
from neighbors_and_roc.options import Data_Defaults
from neighbors_and_roc import my_models
from neighbors_and_roc.options import Model_Defaults
from neighbors_and_roc import util_emb

# SETUP
random.seed(1)
np.random.seed(1)


# OPTIONS
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 100
print(data_opts, "\n")
model_opts = Model_Defaults()
print(model_opts, "\n")


# DATA
#region data preparation
sentences = util_ROC.get_all_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
# train_text, dev_text, test_text = util_data.split_list_in_3(data_opts.TRAIN_PERCENTAGE, data_opts.DEV_PERCENTAGE, stories)

# prepare the tokenizer on the source text #TODO see what be moved out of this file
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
index_to_word = {index: word for (word, index) in tokenizer.word_index.items()}
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Because tokenizer does not assign 0
print('Vocabulary Size: %d' % vocab_size)
# Convert from text to sequences
sequences = tokenizer.texts_to_sequences(sentences)
print('Total Sequences: %d' % len(sequences))
# pad input sequences
max_sentence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sentence_length, padding='pre')
print('Max Sequence Length: %d' % max_sentence_length)
# split into input and output elements
NUM_GOOD_CHOICES = 2
NUM_BAD_CHOICES = 1

def get_bad_choice_sentence_index(id): #TODO
    return 3

# Get list of choice arrays with correct choices marked
# c1 1 0
# c2 0 1
#
# c1 1 0
# c2 0 1
#
# ...
choices_labels = []
for i in range(0,len(sentences)-NUM_GOOD_CHOICES,5):
    good_choices = sequences[i+1:i+1+NUM_GOOD_CHOICES]
    bad_choices = sequences[get_bad_choice_sentence_index(i)]
    all_choices = np.vstack((good_choices,bad_choices)) #TODO swap vstack/hstack -> stack
    all_labels = np.array([1]*NUM_GOOD_CHOICES + [0]*NUM_BAD_CHOICES).reshape((NUM_GOOD_CHOICES+NUM_BAD_CHOICES,1))
    label1 = to_categorical(random.choice(range(NUM_GOOD_CHOICES)), num_classes=NUM_GOOD_CHOICES+NUM_BAD_CHOICES)\
                    .reshape((NUM_GOOD_CHOICES+NUM_BAD_CHOICES, 1))
    label2 = all_labels-label1
    choices_labels.append(np.hstack((all_choices, label1, label2)))

# Shuffle each sample's choices independently
for sample in choices_labels:
    np.random.shuffle(sample)

# Get view of X
# x c1 c2
# x c1 c2
# ...
C = np.vstack([sample[:,:-2].reshape(sample.size-NUM_GOOD_CHOICES*(NUM_GOOD_CHOICES+NUM_BAD_CHOICES)) for sample in choices_labels])
X = np.hstack((sequences[0::5,:],C))
# Get view of Y
# 0 1
# 1 0
# ...
Y = np.vstack([sample[:,-2].reshape(NUM_GOOD_CHOICES+NUM_BAD_CHOICES) for sample in choices_labels])
Y_alternatives = np.vstack([sample[:,-1].reshape(NUM_GOOD_CHOICES+NUM_BAD_CHOICES) for sample in choices_labels])
# print([[index_to_word[i] for i in row if i!=0] for row in X]) # HELPFUL FOR DEBUGGING
#endregion
training_data = (X[:model_opts.BASE_NUM_TRAINING_SAMPLES],
                 Y[:model_opts.BASE_NUM_TRAINING_SAMPLES])
NUM_SAMPLES_TO_ADD = int(model_opts.BASE_NUM_TRAINING_SAMPLES * model_opts.BASE_NUM_TRAINING_SAMPLES)
fresh_training_data = (X[model_opts.BASE_NUM_TRAINING_SAMPLES:model_opts.BASE_NUM_TRAINING_SAMPLES + NUM_SAMPLES_TO_ADD],
                       Y[model_opts.BASE_NUM_TRAINING_SAMPLES:model_opts.BASE_NUM_TRAINING_SAMPLES + NUM_SAMPLES_TO_ADD])
alternative_training_data = (X[:NUM_SAMPLES_TO_ADD],
                             Y_alternatives[:NUM_SAMPLES_TO_ADD])
testing_data = (X[len(X)-model_opts.NUM_TESTING_SAMPLES:],
                Y[len(X)-model_opts.NUM_TESTING_SAMPLES:]+Y_alternatives[len(X)-model_opts.NUM_TESTING_SAMPLES:])


# GET EMBEDDINGS
print("Retrieving embeddings...")
if model_opts.EMBEDDINGS_FILENAME:
    embedding_matrix = util_emb.get_embedding_matrix(
        embedding_size=model_opts.EMBEDDING_SIZE, embedding_path=model_opts.EMBEDDINGS_FILEPATH, vocab_size=vocab_size, word_to_index=tokenizer.word_index)


### TRAIN WITH FRESH DATA ADDED ###
print("Building model...")
if model_opts.EMBEDDINGS_FILENAME:
    model = my_models.model(input_length=X.shape[1], num_classes=NUM_GOOD_CHOICES+NUM_BAD_CHOICES,
                            embedding_matrix=embedding_matrix, hidden_layers=model_opts.HIDDEN_LAYERS)
else:  # Random embedding, mostly for debugging
    model = my_models.model(input_length=X.shape[1], num_classes=NUM_GOOD_CHOICES+NUM_BAD_CHOICES,
                            embedding_matrix_shape=(vocab_size, model_opts.EMBEDDING_SIZE), hidden_layers=model_opts.HIDDEN_LAYERS)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model...")
history = model.fit(np.vstack((training_data[0],fresh_training_data[0])), np.vstack((training_data[1],fresh_training_data[1])),
          validation_data=(testing_data[0], testing_data[1]),
          epochs=model_opts.EPOCHS, batch_size=model_opts.BATCH_SIZE, verbose=0)
#region printing during training
history_log = history.history
epoch=0
for acc in history_log["acc"]:
    print("Epoch {epoch:3}: Loss={loss:.3f} / {val_loss:.3f}  Acc={acc:.3f} / {val_acc:.3f}".
          format(epoch=epoch + 1,
                 loss=history.history["loss"][epoch],
                 acc=acc, val_loss=history.history["val_loss"][epoch],
                 val_acc=history.history["val_acc"][epoch]
                 )
          )
    epoch += 1
#endregion
print("Evaluating model...")
_,acc = model.evaluate(x=testing_data[0],y=testing_data[1],batch_size=model_opts.BATCH_SIZE, verbose=2)
print(acc)


### TRAIN WITH ALTERNATIVE DATA ADDED ###
print("Building model...")
if model_opts.EMBEDDINGS_FILENAME:
    model = my_models.model(input_length=X.shape[1], num_classes=NUM_GOOD_CHOICES+NUM_BAD_CHOICES,
                            embedding_matrix=embedding_matrix, hidden_layers=model_opts.HIDDEN_LAYERS)
else:  # Random embedding, mostly for debugging
    model = my_models.model(input_length=X.shape[1], num_classes=NUM_GOOD_CHOICES+NUM_BAD_CHOICES,
                            embedding_matrix_shape=(vocab_size, model_opts.EMBEDDING_SIZE), hidden_layers=model_opts.HIDDEN_LAYERS)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model...")
history = model.fit(np.vstack((training_data[0],alternative_training_data[0])), np.vstack((training_data[1],alternative_training_data[1])),
          validation_data=(testing_data[0], testing_data[1]),
          epochs=model_opts.EPOCHS, batch_size=model_opts.BATCH_SIZE, verbose=0)
#region printing during training
history_log = history.history
epoch=0
for acc in history_log["acc"]:
    print("Epoch {epoch:3}: Loss={loss:.3f} / {val_loss:.3f}  Acc={acc:.3f} / {val_acc:.3f}".
          format(epoch=epoch + 1,
                 loss=history.history["loss"][epoch],
                 acc=acc, val_loss=history.history["val_loss"][epoch],
                 val_acc=history.history["val_acc"][epoch]
                 )
          )
    epoch += 1
#endregion
print("Evaluating model...")
_,acc = model.evaluate(x=testing_data[0],y=testing_data[1],batch_size=model_opts.BATCH_SIZE, verbose=2)
print(acc)
# TODO make sure evaluation is really allowing both right answers to count as correct

# # TEST
# # for (sentence, x,y) in zip(dev_text, X_dev, Y_dev):
# for i in range(len(dev_text)):
#     sentence = dev_text[i]
#     x = np.reshape(X_dev[i], (1, max_sentence_length - 1))
#     y = Y_dev[i]
#     pred = model.predict_classes(x, batch_size=model_opts.BATCH_SIZE, verbose=2)[0]
#     print("{} -> {}".format(sentence,index_to_word[pred]))


# print(generate_seq(model, tokenizer, max_length - 1, 'Jack', 4))
# print(generate_seq(model, tokenizer, max_length - 1, 'Jill', 4))


# # generate a sequence from a language model
# def generate_seq(model, tokenizer, max_length, seed_text, n_words):
#     in_text = seed_text
#     # generate a fixed number of words
#     for _ in range(n_words):
#         # encode the text as integer
#         encoded = tokenizer.texts_to_sequences([in_text])[0]
#         # pre-pad sequences to a fixed length
#         encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
#         # predict probabilities for each word
#         yhat = model.predict_classes(encoded, verbose=0)
#         # map predicted word index to word
#         out_word = ''
#         for word, index in tokenizer.word_index.items():
#             if index == yhat:
#                 out_word = word
#                 break
#         # append to input
#         in_text += ' ' + out_word
#     return in_text