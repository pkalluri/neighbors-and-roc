# File adapted from https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import random
import os

from neighbors_and_roc import util_ROC, util_emb
from neighbors_and_roc.options import Data_Defaults, Model_Defaults
from neighbors_and_roc import my_models, fixed_settings


# SETUP
# random.seed(1)
# np.random.seed(1)


# OPTIONS
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 15000
print(data_opts, "\n")
model_opts = Model_Defaults()
model_opts.EMBEDDINGS_FILENAME = 'GoogleNews-vectors-negative300.txt'
# model_opts.EMBEDDINGS_FILENAME = None # For quick debugging
model_opts.EMBEDDINGS_FILEPATH = os.path.join(fixed_settings.EMBEDDINGS_ROOT, model_opts.EMBEDDINGS_FILENAME) if model_opts.EMBEDDINGS_FILENAME != None else None
model_opts.HIDDEN_LAYERS = []
# self.BATCH_SIZE = 32
model_opts.EPOCHS = 20
model_opts.DROPOUT = True
model_opts.REGULARIZE = True
model_opts.BASE_NUM_TRAINING_SAMPLES = 5000
model_opts.PERCENTAGE_TO_ADD = 0
model_opts.NUM_TESTING_SAMPLES = 500
model_opts.USING_ALTERNATIVES = False
model_opts.NUM_GOOD_CHOICES = 1
model_opts.NUM_BAD_CHOICES = 4
model_opts.USE_SELF_AS_CHOICE = False
# model_opts = Model_Defaults() # For quick debugging only
print(model_opts, "\n")
assert(data_opts.NUM_STORIES >
       model_opts.BASE_NUM_TRAINING_SAMPLES + model_opts.PERCENTAGE_TO_ADD*model_opts.BASE_NUM_TRAINING_SAMPLES + model_opts.NUM_TESTING_SAMPLES)


# DATA
#region data preparation
sentences = util_ROC.get_all_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)

# prepare the tokenizer on the source text #TODO see what can be moved out of this file
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
index_to_word = {index: word for (word, index) in tokenizer.word_index.items()}
# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Because tokenizer does not assign 0
print('Vocabulary Size: %d' % vocab_size)
# Convert from text to sequences
sequences = tokenizer.texts_to_sequences(sentences)
# pad input sequences
max_sentence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sentence_length, padding='pre')
print('Max Sequence Length: %d' % max_sentence_length)
# split into input and output elements

def get_bad_choice_sentence_index(id): #TODO
    r = random.randint(0,len(sentences)-1)
    return r

# Get list of choice arrays with correct choices marked
# c1 1 0
# c2 0 1
#
# c1 1 0
# c2 0 1
#
# ...
choices_labels = []
for i in range(0,len(sentences)-model_opts.NUM_GOOD_CHOICES,5):
    if model_opts.USE_SELF_AS_CHOICE: good_choices = sequences[i:i+model_opts.NUM_GOOD_CHOICES]
    else: good_choices = sequences[i+1:i+1+model_opts.NUM_GOOD_CHOICES]
    bad_choices = np.vstack(([sequences[get_bad_choice_sentence_index(i),:] for _ in range(model_opts.NUM_BAD_CHOICES)]))
    all_choices = np.vstack((good_choices,bad_choices)) #TODO swap vstack/hstack -> stack
    all_labels = np.array([1]*model_opts.NUM_GOOD_CHOICES + [0]*model_opts.NUM_BAD_CHOICES).reshape((model_opts.NUM_GOOD_CHOICES+model_opts.NUM_BAD_CHOICES,1))
    individual_labels = []
    label1 = to_categorical(random.choice(range(model_opts.NUM_GOOD_CHOICES)), num_classes=model_opts.NUM_GOOD_CHOICES+model_opts.NUM_BAD_CHOICES)\
                    .reshape((model_opts.NUM_GOOD_CHOICES+model_opts.NUM_BAD_CHOICES, 1))
    individual_labels.append(label1)
    sum_labels_so_far = label1
    for remaining_choice in range(model_opts.NUM_GOOD_CHOICES-1):
        label = all_labels - sum_labels_so_far
        individual_labels.append(label)
        sum_labels_so_far = sum_labels_so_far + label
    individual_labels = np.hstack(individual_labels)
    choices_labels.append(np.hstack((all_choices, individual_labels)))

# Shuffle each sample's choices independently
for sample in choices_labels:
    np.random.shuffle(sample)

# Get view of X
# x c1 c2
# x c1 c2
# ...
C = np.vstack([sample[:,:-1*model_opts.NUM_GOOD_CHOICES].reshape(sample.size-model_opts.NUM_GOOD_CHOICES*(model_opts.NUM_GOOD_CHOICES+model_opts.NUM_BAD_CHOICES)) for sample in choices_labels])
X = np.hstack((sequences[0::5,:],C))
# Get view of Y
# 0 1
# 1 0
# ...
# TODO shuffle before train/test split
Y = np.vstack([sample[:,-1*model_opts.NUM_GOOD_CHOICES].reshape(model_opts.NUM_GOOD_CHOICES+model_opts.NUM_BAD_CHOICES) for sample in choices_labels])
Y_alternatives = np.vstack([sample[:,-1].reshape(model_opts.NUM_GOOD_CHOICES+model_opts.NUM_BAD_CHOICES) for sample in choices_labels])
# print([[index_to_word[i] for i in row if i!=0] for row in X]) # HELPFUL FOR DEBUGGING
#endregion
basic_training_data = (X[:model_opts.BASE_NUM_TRAINING_SAMPLES],
                 Y[:model_opts.BASE_NUM_TRAINING_SAMPLES])
NUM_SAMPLES_TO_ADD = int(model_opts.BASE_NUM_TRAINING_SAMPLES * model_opts.PERCENTAGE_TO_ADD)
fresh_training_data = (X[model_opts.BASE_NUM_TRAINING_SAMPLES:model_opts.BASE_NUM_TRAINING_SAMPLES + NUM_SAMPLES_TO_ADD],
                       Y[model_opts.BASE_NUM_TRAINING_SAMPLES:model_opts.BASE_NUM_TRAINING_SAMPLES + NUM_SAMPLES_TO_ADD])
alternative_training_data = (X[:NUM_SAMPLES_TO_ADD],
                             Y_alternatives[:NUM_SAMPLES_TO_ADD])
testing_data = (X[len(X)-model_opts.NUM_TESTING_SAMPLES:],
                Y[len(X)-model_opts.NUM_TESTING_SAMPLES:]+Y_alternatives[len(X)-model_opts.NUM_TESTING_SAMPLES:])


# GET EMBEDDINGS
if model_opts.EMBEDDINGS_FILENAME:
    print("Retrieving embeddings...")
    embedding_matrix = util_emb.get_embedding_matrix(
        embedding_size=model_opts.EMBEDDING_SIZE, embedding_path=model_opts.EMBEDDINGS_FILEPATH, vocab_size=vocab_size, word_to_index=tokenizer.word_index)


# TRAIN AND EVALUATE
additional_training_data = alternative_training_data if model_opts.USING_ALTERNATIVES else fresh_training_data
print("Building model...")
if model_opts.EMBEDDINGS_FILENAME:
    model = my_models.classifier_model(input_length=X.shape[1], num_classes=model_opts.NUM_GOOD_CHOICES + model_opts.NUM_BAD_CHOICES,
                                       embedding_matrix=embedding_matrix,
                                       hidden_layers=model_opts.HIDDEN_LAYERS, dropout=model_opts.DROPOUT, regularize=model_opts.REGULARIZE)
else:  # Random embedding, mostly for debugging
    model = my_models.classifier_model(input_length=X.shape[1], num_classes=model_opts.NUM_GOOD_CHOICES + model_opts.NUM_BAD_CHOICES,
                                       embedding_matrix_shape=(vocab_size, model_opts.EMBEDDING_SIZE),
                                       hidden_layers=model_opts.HIDDEN_LAYERS, dropout=model_opts.DROPOUT, regularize=model_opts.REGULARIZE)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Training model...")
history = model.fit(np.vstack((basic_training_data[0], additional_training_data[0])), np.vstack((basic_training_data[1], additional_training_data[1])),
                    validation_data=(testing_data[0], testing_data[1]),
                    epochs=model_opts.EPOCHS, batch_size=model_opts.BATCH_SIZE, verbose=2)
print("Evaluating model...")
#region evaluate
# Note that model.evaluate() is intentionally not used here because it will only mark 1 (the first) of the 2 valid answers correct
num_correct = 0
for i in range(model_opts.NUM_TESTING_SAMPLES):
    x,y = testing_data[0][i,:].reshape((1,testing_data[0].shape[1])), testing_data[1][i,:].reshape((1,testing_data[1].shape[1]))
    prediction = model.predict_classes(x)[0]
    if y[0,prediction] >= 1: print('\nCorrect:'); num_correct += 1
    else: print('\nWrong:')
    choice_num = 0
    for start in range(0,x.shape[1],max_sentence_length):
        predicted_flag = ''
        gold_star_flag = ''
        if start != 0:
            if choice_num == prediction: predicted_flag = '> '
            if y[0,choice_num] >= 1: gold_star_flag = '* '
            choice_num += 1
        print(gold_star_flag+predicted_flag+
              ' '.join([index_to_word[i] for i in x[0,start:start+max_sentence_length] if i!=0])) # HELPFUL FOR DEBUGGING
        if start == 0:
            print('--')
print('\n{correct}/{all}={perc:.2%}'.format(correct=num_correct, all=model_opts.NUM_TESTING_SAMPLES,
                                        perc=num_correct/model_opts.NUM_TESTING_SAMPLES))
#endregion