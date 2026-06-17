import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.empty(capacity, dtype=object)
        self.write_ptr = 0 # where next experience is written
        self.n_entries = 0 # current buffer size

    # update parent's sum after leaf changes
    def propagate(self, idx, change):
        parent = (idx-1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    # to find the correct leaf -> basically binary search
    def retrieve(self, idx, sample_val):
        left = 2*idx+1
        right = left+1
        if left >= len(self.tree):
            return idx
        if sample_val <= self.tree[left]:
            return self.retrieve(left, sample_val)
        return self.retrieve(right, sample_val - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries+1,self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self.propagate(idx, change)

    def get(self,sample_val):
        idx = self.retrieve(0, sample_val)
        data_idx = idx-self.capacity+1
        return idx, self.tree[idx],self.data[data_idx]



