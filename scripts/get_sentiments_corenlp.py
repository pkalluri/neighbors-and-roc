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
WRITE_PATH = os.path.join(fixed_settings.GENERATED_DATA_ROOT,'ROC-to-sentiment-coreNLP.txt')

# Sentences
stories, contexts, completions, sentences = util_ROC.get_stories_contexts_completions_and_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
sentences = completions

# Sentiment
proc = Popen('java -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -stdin', shell=True, stdin=PIPE, stdout=PIPE)
out,err = proc.communicate('\n'.join(sentences).encode('utf-8'))
label_words = out.decode('utf-8').split('\n')
print(label_words)

translate_to_symbols = {}
translate_to_symbols['Positive'] = '+'
translate_to_symbols['Very positive'] = '+'
translate_to_symbols['Negative'] = '-'
translate_to_symbols['Very negative'] = '-'
translate_to_symbols['Neutral'] = 'n'
labels = [translate_to_symbols[label_word.strip()] for label_word in label_words if label_word.strip()!='']

sentiment_to_simple_responses = {}
sentiment_to_simple_responses['+']=['That was good.', 'That was great.']
sentiment_to_simple_responses['-']=['That was bad.', 'That was terrible.']
dataset = [label.strip()+' '+x+' '+random.choice(sentiment_to_simple_responses[label.strip()])
           for (label,x) in zip(labels,X) if label.strip() in sentiment_to_simple_responses] # Filter out neutrals
print('\n'.join(dataset), file=open(WRITE_PATH, 'w'))

# with open(os.path.join(fixed_settings.GENERATED_DATA_ROOT,'completions_sentiments-100.txt'), 'w') as f:
#     f.write('\n'.join(sentiment_labels))
#     f.close()
