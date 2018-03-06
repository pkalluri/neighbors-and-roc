import json
import os
from neighbors_and_roc import fixed_settings

"""To set options"""
class Options(object):
    def __init__(self):
        return
    def __str__(self):
        return json.dumps(vars(self), indent=3)

class Data_Defaults(Options):
    def __init__(self):
        # Data
        self.ROC_FILENAME = 'ROCStories_winter2017.csv'
        self.ROC_FILEPATH = os.path.join(fixed_settings.DATA_ROOT, self.ROC_FILENAME)
        self.NUM_STORIES = 3
        self.SENTIMENTS_FILEPATH = None # Set this explicitly and set this last

class Model_Defaults(Options):
    def __init__(self):
        self.EMBEDDINGS_FILENAME = 'GoogleNews-vectors-negative300.txt'
        self.EMBEDDINGS_FILENAME = None
        self.EMBEDDINGS_FILEPATH = os.path.join(fixed_settings.EMBEDDINGS_ROOT, self.EMBEDDINGS_FILENAME) if self.EMBEDDINGS_FILENAME!=None else None
        self.EMBEDDING_SIZE = 300

        self.HIDDEN_LAYERS = [150,75]
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.BASE_NUM_TRAINING_SAMPLES = 10000
        self.PERCENTAGE_TO_ADD = 0
        self.NUM_TESTING_SAMPLES = 1000