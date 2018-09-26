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
				# 'n_filters_CNN': 50,
				'n_filters_CNN': 100,
				'kernel_size': 10,
				'P': 10,
				# 'n_filters_RNN_list': [50, 1],
				'n_filters_RNN_list': [50, 50],
				'alpha': 1.0
				})

		ae_ckpt_path = os.path.join('ae_ckpt','model.ckpt')

		saver = tf.train.Saver(var_list=tf.trainable_variables())
		with tf.Session() as sess:
			saver.restore(sess, ae_ckpt_path)
			init_decoded_list = sess.run(model.auto_encoder.decoder, \
							feed_dict={model.auto_encoder.input_: self.data.seq_list, model.auto_encoder.input_batch_size:len(self.data.seq_list)})

		dc_ckpt_path = os.path.join('dtc_ckpt','model.ckpt')
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, dc_ckpt_path)
			decoded_list, self.cluster_list, self.z_list = sess.run([model.auto_encoder.decoder, model.pred, model.z], \
							feed_dict={model.auto_encoder.input_: self.data.seq_list, model.auto_encoder.input_batch_size:len(self.data.seq_list)})

		self.decoded_list_list = [init_decoded_list, decoded_list]


	def plot_decoded_sequences(self, stop_idx=None):

		savedir = os.path.join('.','_fig')
		if os.path.exists(savedir):
			shutil.rmtree(savedir)
		os.mkdir(savedir)

		print('writing ...')
		N = len(self.data.seq_list) if stop_idx == None else stop_idx
		p = ProgressBar(maxval=N).start()
		for idx in range(N):
			_dir = 'dat{}'.format(idx)
			os.mkdir(os.path.join(savedir,_dir))

			fig, ax = plt.subplots(nrows=1+len(self.decoded_list_list), ncols=1, figsize=(20,15))
			fig.suptitle('[{}] upper: original sequence, lower: decoded sequence'.format(_dir), fontsize=10)
			plt.subplots_adjust(hspace=0.2)

			X = self.data.seq_list[idx]
			ax[0].plot(X)
			for i, decoded_list in enumerate(self.decoded_list_list):
				decoded = decoded_list[idx]
				ax[i+1].plot(decoded)
			plt.savefig(os.path.join(savedir,_dir,'result.png'))
			plt.close()

			# print(_dir, seg_list, cluster_list[idx*4:(idx+1)*4])
			with open(os.path.join(savedir,_dir,'cluster.txt'), 'w') as fo:
				fo.write('{}'.format(self.cluster_list[idx]))

			if idx == N-1:
				break

			p.update(idx+1)
		p.finish()

def main():
	args = generate_args()

	ilm = InferenceLearnedModel(args)
	ilm.plot_decoded_sequences(stop_idx=1)

	params = {'dimensions':2,
			'perplexity':30.0,
			'theta':0.5,
			'rand_seed':-1}
	bhtsne = BHTSNE(params)
	bhtsne.fit_and_plot(ilm.z_list.reshape((len(ilm.z_list),-1)), ilm.data.label_list, ilm.cluster_list)

if __name__ == '__main__':
	main()