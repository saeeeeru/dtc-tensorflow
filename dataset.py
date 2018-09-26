import os, random, math

import numpy as np
import matplotlib.pyplot as plt

def shuffle_samples(X, y):
	zipped = list(zip(X, y))
	np.random.shuffle(zipped)
	X_result, y_result = zip(*zipped)
	return np.asarray(X_result), np.asarray(y_result)

# Please edit following code for your purpose
def prepare_dataset(K):
	n_clusters, N, L, dt = 4, 150, 100, 0.1
	t = np.arange(0, L*dt, dt)[:L]
	seq_list, label_list = [], []
	for i in range(n_clusters):
		n_sinusoids = np.random.random_integers(1,4)
		sample_parameters = [[np.random.normal(loc=1, scale=2, size=K), np.random.normal(loc=10, scale=5, size=K)] for _ in range(n_sinusoids)]
		for j in range(N):
			seq = np.vstack([np.sum([coef[k]*np.sin(2*np.pi*freq[k]*t) for coef, freq in sample_parameters], axis=0) + np.random.randn(L) for k in range(K)]).reshape(L,K)
			seq_list.append(seq); label_list.append(i)

	return seq_list, label_list

class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, dataset):
		self.K = 3
		if dataset == 'synthetic':
			seq_list, label_list = prepare_dataset(self.K)
		else: 
			assert False, 'does not exists dataset: {}.'.format(dataset)

		self.L = seq_list[0].shape[0]

		# shuffle and split dataset to training and test
		self.seq_list, self.label_list = shuffle_samples(seq_list, label_list)
		n_training = int(len(self.seq_list)*0.8)
		self.train_seq, self.train_label = np.array(self.seq_list[:n_training]), self.label_list[:n_training]
		self.test_seq, self.test_label = np.array(self.seq_list[n_training:]), self.label_list[n_training:]
		print('dataset size: train={}, test={}'.format(len(self.train_seq),len(self.test_seq)))

	def gen_next_batch(self, batch_size, is_train_set=True, epoch=None, iteration=None):
		if is_train_set == True:
			x = self.train_seq
			y = self.train_label
		else:
			x = self.test_seq
			y = self.test_label

		assert len(x)>=batch_size, "batch size must be smaller than data size: {}.".format(len(x))

		if epoch != None:
			until = math.ceil(float(epoch*len(x))/float(batch_size))
		elif iteration != None:
			until = iteration
		else:
			assert False, "epoch or iteration must be set."

		iter_ = 0
		index_list = [i for i in range(len(x))]
		while iter_ <= until:
			idxs = random.sample(index_list, batch_size)
			iter_ += 1
			yield (x[idxs], y[idxs], idxs)

if __name__ == '__main__':
	Dataset()