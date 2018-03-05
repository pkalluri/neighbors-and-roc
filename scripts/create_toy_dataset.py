from neighbors_and_roc.options import Data_Defaults
from neighbors_and_roc import fixed_settings

import os

# SETTINGS
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 100
data_opts.SENTIMENTS_FILEPATH = os.path.join(fixed_settings.GENERATED_DATA_ROOT,
                                             'first_sentences_sentiments-'+str(data_opts.NUM_STORIES)+'.txt')

from neighbors_and_roc import util_ROC, fixed_settings
import os
import random

# DATA
stories, contexts, completions = util_ROC.get_stories_contexts_and_completions(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
X = [context[0] for context in contexts]
Y = []
simple_positive_sentences = ['That felt good.', 'That was great.']
simple_negative_sentences = ['That felt bad.', 'That was terrible.']
with open(os.path.join(fixed_settings.GENERATED_DATA_ROOT, data_opts.SENTIMENTS_FILEPATH), 'r') as file:
    for i in range(data_opts.NUM_STORIES):
        sentiment_label = file.readline().strip()
        if sentiment_label=='Positive':
            Y.append(random.choice(simple_positive_sentences))
        if sentiment_label=='Negative':
            Y.append(random.choice(simple_negative_sentences))
        if sentiment_label=='Neutral':
            Y.append('')
dataset = [pair for pair in zip(X,Y) if pair[1]!=''] # Filter out neutrals
print('\n'.join([' '+' '.join(x) for x in dataset]), file=open('tmp.txt','w'))

# PREPARE DATA
from neighbors_and_roc import util_ROC,my_models,util_emb,util_data
from neighbors_and_roc.options import Model_Defaults
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from random import shuffle