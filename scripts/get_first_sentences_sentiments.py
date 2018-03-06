import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))

from neighbors_and_roc import util_ROC, fixed_settings
from neighbors_and_roc.options import Data_Defaults

from subprocess import Popen, PIPE
import random
import os

# Settings
data_opts = Data_Defaults()
data_opts.NUM_STORIES = 100 # All stories

# Sentences
stories, contexts, completions = util_ROC.get_stories_contexts_and_completions(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
first_sentences = [completion[0] for completion in completions]

# Sentiment
proc = Popen('java -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -stdin', shell=True, stdin=PIPE, stdout=PIPE)
out,err = proc.communicate('\n'.join(first_sentences).encode('utf-8'))
sentiment_labels = out.decode('utf-8').split()

# Write
with open(os.path.join(fixed_settings.GENERATED_DATA_ROOT,'completions_sentiments-100.txt'), 'w') as f:
    f.write('\n'.join(sentiment_labels))
    f.close()

# simple_positive_sentences = ['That felt good.', 'That was great.']
# simple_negative_sentences = ['That felt bad.', 'That was terrible.']
# simple_next_sentences =[random.choice(simple_positive_sentences) if sentiment_label=='Positive'
#                         else random.choice(simple_negative_sentences)
#                         for sentiment_label in sentiment_labels]
# print(zip(first_sentences, simple_next_sentences))

# from httplib2 import Http
# from urllib.parse import urlencode
# import json
# from textblob import TextBlob
# sentiment_analyzer_url = 'http://text-processing.com/api/sentiment/'
# h = Http()
# counter = 0
# for sentence in first_sentences:
#     # print(sentence)
#
#     # _,content = h.request(sentiment_analyzer_url, "POST", urlencode({'text':sentence}))
#     # json_str = content.decode('utf-8')
#     # json_dict = json.loads(json_str)
#     # label = json_dict['label']
#     # if label != 'neutral':
#     #     print('{0} {1} neutral:{3:.2%} {2}:{4:.2%}'.format(counter, sentence, label, json_dict['probability']['neutral'], json_dict['probability'][label]))
#     #     counter += 1
#
#     _,content = h.request('http://nlp.stanford.edu:8080/sentiment/rntnDemo.html', "POST", urlencode({'text':sentence}))
#     print(content)
#
#
#     # s = TextBlob(sentence)
#     # print('{sentence} {value:.2%}'.format(sentence=sentence, value=s.sentiment.polarity))
