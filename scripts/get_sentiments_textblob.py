from neighbors_and_roc.options import Data_Defaults
from neighbors_and_roc import util_ROC, fixed_settings

from httplib2 import Http
from urllib.parse import urlencode
import json
from textblob import TextBlob
import os
import random

# SETTINGS
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 1000
sentiment_analyzer_url = 'http://text-processing.com/api/sentiment/'
WRITE_PATH = os.path.join(fixed_settings.GENERATED_DATA_ROOT,'ROC-to-sentiment-textblob.txt')

# DATA
stories, contexts, completions, sentences = util_ROC.get_stories_contexts_completions_and_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
X = completions

labels = []
for sentence in X:
    s = TextBlob(sentence)
    polarity = s.sentiment.polarity
    if polarity>0: labels.append('+')
    if polarity<0: labels.append('-')
    if polarity==0: labels.append('n')

sentiment_to_simple_responses = {}
sentiment_to_simple_responses['+']=['That was good.', 'That was great.']
sentiment_to_simple_responses['-']=['That was bad.', 'That was terrible.']
dataset = [label+' '+x+' '+random.choice(sentiment_to_simple_responses[label])
           for (label,x) in zip(labels,X) if label in sentiment_to_simple_responses] # Filter out neutrals
print('\n'.join(dataset), file=open(WRITE_PATH, 'w'))