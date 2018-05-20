import numpy as np

class CountTracker(object):
    def __init__(self, granularity):
        self.granularity = granularity
        self.hyperplanes = np.random.normal(size=(self.granularity, 3))
        #maps hashcodes to number of visits
        self.hashtable = {}
        #maps goals to hashcodes
        self.goal_hash = {}

    def compute_hash_code(self, state):
        result = np.dot(self.hyperplanes, state)

        string_version = ""
        #Apply sgn operation to each scalar and placing in a binary string
        for i in range(len(result)):
            if(result[i] >= 0):
                string_version += "1"
            else:
                string_version += "0"
        hash_code = int(string_version)
        return hash_code

    def update_count(self, hash_code):
        if hash_code in self.hashtable:
            self.hashtable[hash_code] += 1
        else:
            self.hashtable[hash_code] = 1
