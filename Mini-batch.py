import numpy as np
import math
def random_minibatches ( batch_size , X , Y):
        m = X.shape[1]
        mini_batches = []
        permutations = np.random.permutation(m)

        shuffled_X = X[: , permutations]
        shuffled_Y = Y[: , permutations]

        num_of_completed_batches = math.floor(m / batch_size)
        for k in range ( 1 , num_of_completed_batches + 1):
             mini_batch_X = X [ : , (k-1) * batch_size : k * batch_size]
             mini_batch_Y = Y [ : , (k-1) * mini_batch_size : (k) * mini_batch_size]
             mini_batch = (mini_batch_X , mini_batch_Y)
             mini_batches.append(mini_batch)

        if m % batch_size != 0 :
            mini_batch_X = X [ : , num_of_completed_batches * batch_size : m]
            mini_batch_Y = Y [ : , num_of_completed_batches* batch_size : m ]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches
