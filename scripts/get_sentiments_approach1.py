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
data_opts.NUM_STORIES = 100
sentiment_analyzer_url = 'http://text-processing.com/api/sentiment/'
WRITE_PATH = os.path.join(fixed_settings.GENERATED_DATA_ROOT,'ROC-to-sentiment-1.txt')

# DATA
stories, contexts, completions, sentences = util_ROC.get_stories_contexts_completions_and_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
X = completions

labels = []
h = Http()
for sentence in X:
    _,content = h.request(sentiment_analyzer_url, "POST", urlencode({'text':sentence}))
    json_str = content.decode('utf-8')
    json_dict = json.loads(json_str)
    label = json_dict['label']
    if label=='pos': labels.append('+')
    if label=='neg': labels.append('-')
    if label=='neutral': labels.append('n')

sentiment_to_simple_responses = {}
sentiment_to_simple_responses['+']=['That was good.', 'That was great.']
sentiment_to_simple_responses['-']=['That was bad.', 'That was terrible.']
dataset = [label+' '+x+' '+random.choice(sentiment_to_simple_responses[label])
           for (label,x) in zip(labels,X) if label in sentiment_to_simple_responses] # Filter out neutrals
print('\n'.join(dataset), file=open(WRITE_PATH, 'w'))