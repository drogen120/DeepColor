"""DataSet base class
"""

class DataSet(object):
    def __init__(self, common_params, dataset_params):
        raise NotImplementedError

    def batch(self):
        #Get batch
        raise NotImplementedError
