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

class Model_Defaults(Options):
    def __init__(self):
        # self.EMBEDDINGS_FILENAME = 'GoogleNews-vectors-negative300.txt'
        self.EMBEDDINGS_FILENAME = 'GoogleNews-vectors-negative300.txt'
        # self.EMBEDDINGS_FILENAME = None
        self.EMBEDDINGS_FILEPATH = os.path.join(fixed_settings.EMBEDDINGS_ROOT, self.EMBEDDINGS_FILENAME) if self.EMBEDDINGS_FILENAME!=None else None
        self.EMBEDDING_SIZE = 300

        self.HIDDEN_LAYERS = []
        self.BATCH_SIZE = 64
        self.EPOCHS = 2
        self.DROPOUT = True
        self.REGULARIZE = True
        self.BASE_NUM_TRAINING_SAMPLES = 10
        self.PERCENTAGE_TO_ADD = 0
        self.NUM_TESTING_SAMPLES = 10

        self.USING_ALTERNATIVES = True