import os, sys, shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from progressbar import ProgressBar

from dataset import Dataset
from dtc.model import DeepTemporalClustering
from func import generate_args, BHTSNE

class InferenceLearnedModel():
	"""docstring for InferenceLearnedModel"""
	def __init__(self, args):
		self.__dict__ = args.copy()

		self.data = Dataset(self.dataset)

		model = DeepTemporalClustering(params={
				'n_clusters': 4,
				'L': self.data.L,
				'K': self.data.K,
				'n_filters_CNN': 100,
				'kernel_size': 10,
				'P': 10,
				'n_filters_RNN_list': [50, 50],
				'alpha': 1.0
				})

		ae_ckpt_path = os.path.join('ae_ckpt', 'model')

		# Build model with sample input
		sample_input = tf.constant(self.data.seq_list[:1], dtype=tf.float32)
		_ = model.auto_encoder(sample_input, 1)

		# Load AE weights and run inference
		model.auto_encoder.load_weights(ae_ckpt_path)
		seq_tensor = tf.constant(self.data.seq_list, dtype=tf.float32)
		init_encoded, init_decoded_list = model.auto_encoder(seq_tensor, len(self.data.seq_list))
		init_decoded_list = init_decoded_list.numpy()

		# Load DTC weights
		dtc_ckpt_path = os.path.join('dtc_ckpt', 'model')
		model.auto_encoder.load_weights(dtc_ckpt_path)

		# Load mu
		mu_path = os.path.join('dtc_ckpt', 'mu.npy')
		mu_values = np.load(mu_path)
		model.initialize_mu(mu_values.shape)
		model.mu.assign(mu_values)

		# Run inference
		encoded, decoded_list = model.auto_encoder(seq_tensor, len(self.data.seq_list))
		decoded_list = decoded_list.numpy()

		q = model.soft_assignment(encoded, model.mu)
		self.cluster_list = tf.argmax(q, axis=1).numpy()
		self.z_list = encoded.numpy()

		self.decoded_list_list = [init_decoded_list, decoded_list]


	def plot_decoded_sequences(self, stop_idx=None):

		savedir = os.path.join('.', '_fig')
		if os.path.exists(savedir):
			shutil.rmtree(savedir)
		os.mkdir(savedir)

		print('writing ...')
		N = len(self.data.seq_list) if stop_idx == None else stop_idx
		p = ProgressBar(maxval=N).start()
		for idx in range(N):
			_dir = 'dat{}'.format(idx)
			os.mkdir(os.path.join(savedir, _dir))

			fig, ax = plt.subplots(nrows=1+len(self.decoded_list_list), ncols=1, figsize=(20, 15), sharex=True, sharey=True)
			fig.suptitle('[{}] upper: original sequence, lower: decoded sequence'.format(_dir), fontsize=10)
			plt.subplots_adjust(hspace=0.2)

			X = self.data.seq_list[idx]
			ax[0].plot(X)
			for i, decoded_list in enumerate(self.decoded_list_list):
				decoded = decoded_list[idx]
				ax[i+1].plot(decoded)
			plt.savefig(os.path.join(savedir, _dir, 'result.png'))
			plt.close()

			with open(os.path.join(savedir, _dir, 'cluster.txt'), 'w') as fo:
				fo.write('{}'.format(self.cluster_list[idx]))

			if idx == N-1:
				break

			p.update(idx+1)
		p.finish()

def main():
	args = generate_args()

	ilm = InferenceLearnedModel(args)
	ilm.plot_decoded_sequences(stop_idx=10)

	params = {'dimensions': 2,
			'perplexity': 30.0,
			'theta': 0.5,
			'rand_seed': -1}

	bhtsne = BHTSNE(params)
	bhtsne.fit_and_plot(ilm.z_list.reshape((len(ilm.z_list), -1)), ilm.data.label_list, ilm.cluster_list)

if __name__ == '__main__':
	main()
