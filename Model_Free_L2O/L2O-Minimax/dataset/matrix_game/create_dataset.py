import numpy as np

dim = 1
p = 1.0
n_train = 256
n_eval = 128
n_test = 128
low = 0.5
high = 1.0

data_train = np.random.uniform(low=low, high=high, size=(n_train, dim, dim))
mask_train = np.random.binomial(n=1, p=p, size=(n_train, dim, dim))
data_train = np.reshape(data_train * mask_train, (n_train, dim * dim))

data_eval = np.random.uniform(low=low, high=high, size=(n_eval, dim, dim))
mask_eval = np.random.binomial(n=1, p=p, size=(n_eval, dim, dim))
data_eval = np.reshape(data_eval * mask_eval, (n_eval, dim * dim))

data_test = np.random.uniform(low=low, high=high, size=(n_test, dim, dim))
mask_test = np.random.binomial(n=1, p=p, size=(n_test, dim, dim))
data_test = np.reshape(data_test * mask_test, (n_test, dim * dim))

np.savetxt('dim{}_p{}_low{}_high{}_train.txt'.format(dim, p, low, high),
           data_train)
np.savetxt('dim{}_p{}_low{}_high{}_eval.txt'.format(dim, p, low, high),
           data_eval)
np.savetxt('dim{}_p{}_low{}_high{}_test.txt'.format(dim, p, low, high),
           data_test)
