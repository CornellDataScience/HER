import numpy as np

class CountTracker(object):
    def __init__(self, granularity):
        self.granularity = granularity
        self.hyperplanes = np.random.normal(size=(self.granularity, 3))
        #maps hashcodes to number of visits
        self.hashtable = {}
        #more compact representation -- redundant now, TODO: Remove
        self.uniqueHashes = {}

    def compute_hash_code(self, state):
        #print("A: ")
        #print(self.hyperplanes)
        #print("state: ")
        #print(state)
        result = np.dot(self.hyperplanes, state)
        #print("result: ")
        #print(result)

        string_version = ""
        #Apply sgn operation to each scalar and placing in a binary string
        for i in range(len(result)):
            if(result[i] >= 0):
                string_version += "1"
            else:
                string_version += "0"
        hash_code = int(string_version, 2)
        if hash_code in self.uniqueHashes:
            self.uniqueHashes[hash_code] += 1
        else:
            self.uniqueHashes[hash_code] = 1
        return hash_code

    def update_count(self, hash_code):
        if hash_code in self.hashtable:
            self.hashtable[hash_code] += 1
        else:
            self.hashtable[hash_code] = 1
