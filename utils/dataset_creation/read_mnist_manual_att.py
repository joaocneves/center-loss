import numpy as np

soft_feat = \
        [[1, 1, 1, 0, 1, 1, 1],
         [0, 0, 0, 0, 0, 1, 1],
         [0, 1, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1],
         [1, 0, 0, 1, 0, 1, 1],
         [1, 0, 1, 1, 1, 0, 1],
         [1, 1, 1, 1, 1, 0, 1],
         [0, 0, 1, 0, 0, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 0, 1, 1, 1, 1, 1]]

soft_feat = np.array(soft_feat).astype('float32')

np.save('datasets\\mnist\\atts_mnist.npy', np.array(soft_feat))
