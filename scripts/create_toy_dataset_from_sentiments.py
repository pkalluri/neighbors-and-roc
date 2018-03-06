from neighbors_and_roc.options import Data_Defaults
from neighbors_and_roc import fixed_settings

import os

# SETTINGS
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 100
data_opts.SENTIMENTS_FILEPATH = os.path.join(fixed_settings.GENERATED_DATA_ROOT,
                                             'completions_sentiments-'+str(data_opts.NUM_STORIES)+'.txt')

from neighbors_and_roc import util_ROC, fixed_settings
import os
import random

# DATA
stories, contexts, completions = util_ROC.get_stories_contexts_completions_and_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
X = completions
Y = []
# simple_positive_sentences = ['That was good.', 'That was great.']
# simple_negative_sentences = ['That was bad.', 'That was terrible.']
with open(os.path.join(fixed_settings.GENERATED_DATA_ROOT, data_opts.SENTIMENTS_FILEPATH), 'r') as file:
    for i in range(data_opts.NUM_STORIES):
        sentiment_label = file.readline().strip()
        if sentiment_label=='Positive':
            # Y.append(random.choice(simple_positive_sentences))
            Y.append(1)
        if sentiment_label=='Negative':
            # Y.append(random.choice(simple_negative_sentences))
            Y.append(-1)
        if sentiment_label=='Neutral':
            Y.append(0)

Z = []
from httplib2 import Http
from urllib.parse import urlencode
import json
from textblob import TextBlob
sentiment_analyzer_url = 'http://text-processing.com/api/sentiment/'
h = Http()
# counter = 0
for sentence in X:
    _,content = h.request(sentiment_analyzer_url, "POST", urlencode({'text':sentence}))
    json_str = content.decode('utf-8')
    json_dict = json.loads(json_str)
    label = json_dict['label']
    # print(label)
    if label=='pos': Z.append(1)
    if label=='neg': Z.append(-1)
    if label=='neutral': Z.append(0)
    # if label != 'neutral':
    #     print('{0} {1} neutral:{3:.2%} {2}:{4:.2%}'.format(counter, sentence, label, json_dict['probability']['neutral'], json_dict['probability'][label]))
    #     counter += 1

    # s = TextBlob(sentence)
    # print('{sentence} {value:.2%}'.format(sentence=sentence, value=s.sentiment.polarity))
# print([(pair[0],pair[1],pair[0]+pair[1]) for pair in zip(Y,Z)])
dataset = [(triplet[0],triplet[1]+triplet[2]) for triplet in zip(X,Y,Z) if triplet[1]!=''] # Filter out neutrals
print('\n'.join([' '+'{} {}'.format(x[0],str(x[1])) for x in dataset]), file=open(data_opts.SENTIMENTS_FILEPATH+'-2ways','w'))

# # PREPARE DATA
# from neighbors_and_roc import util_ROC,my_models,util_emb,util_data
# from neighbors_and_roc.options import Model_Defaults
# import numpy as np
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# from keras.utils import to_categorical
# from random import shuffle