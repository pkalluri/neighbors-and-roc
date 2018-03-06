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
# write_file =

# Sentences
stories, contexts, completions = util_ROC.get_stories_contexts_completions_and_sentences(data_opts.ROC_FILEPATH, num_stories=data_opts.NUM_STORIES)
sentences = completions

# Sentiment
proc = Popen('java -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -stdin', shell=True, stdin=PIPE, stdout=PIPE)
out,err = proc.communicate('\n'.join(sentences).encode('utf-8'))
sentiment_labels = out.decode('utf-8').split()

# Write
with open(os.path.join(fixed_settings.GENERATED_DATA_ROOT,'completions_sentiments-100.txt'), 'w') as f:
    f.write('\n'.join(sentiment_labels))
    f.close()