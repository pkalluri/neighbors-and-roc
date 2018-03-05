import json
import os
from demo_effect_of_neighbors import fixed_settings

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