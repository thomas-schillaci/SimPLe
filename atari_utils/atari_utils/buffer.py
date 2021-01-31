from heapq import heappop, heappush

import torch


class Buffer:

    def __init__(self):
        self.data = []

    def insert(self, experience):
        self.data.append(experience)

    def sample(self, batch_size):
        res = []

        for _ in range(batch_size):
            index = int(torch.randint(0, len(self.data), (1,)))
            sample = self.data[index]
            for i, elem in enumerate(sample):
                if i < len(res):
                    res[i].append(elem)
                else:
                    res.append([elem])

        for i in range(len(res)):
            res[i] = torch.stack(res[i])

        return res


class Buffer_:

    def __init__(self, alpha=0.7, beta=0.5):
        self.alpha = alpha
        self.beta = beta
        self.heap = []
        self.data = {}
        self.n = 0

    def max_priority(self):
        if not self.heap:
            return 1
        return -max(self.heap)[0]

    def insert(self, experience, priority=None):
        if priority is None:
            priority = self.max_priority()
        self.data[self.n] = experience
        heappush(self.heap, (-priority, self.n))
        self.n += 1

    def sample(self, n):
        ranks = torch.arange(len(self.heap)) + 1
        priorities = 1 / ranks
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()
        indices = torch.multinomial(probabilities, n)
        heap_tmp = self.heap.copy()
        experience_indices = [heappop(self.heap) for _ in range(indices.max() + 1)]
        self.heap = heap_tmp
        experience_indices = torch.tensor(experience_indices)[:, 1]
        experience_indices = torch.gather(experience_indices, 0, indices)
        experience = [self.data[int(index)] for index in experience_indices]
        weights = len(self.data) * probabilities
        weights = weights ** (- self.beta)
        weights = weights / weights.max()
        weights = torch.gather(weights, 0, indices)
        return experience, weights, indices

    def update_deltas(self, indices, deltas):
        removed = []
        for _ in range(max(indices) + 1):
            priority, index = heappop(self.heap)
            priority = -priority
            index = self.data.pop(index)
            removed.append([priority, index])
        for i, index in enumerate(indices):
            removed[index][0] = deltas[i]
        for elem in removed:
            self.insert(elem[1], priority=elem[0])
