import os, csv

import configargparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import Dataset
from dtc.model import DeepTemporalClustering

PLT = False

def show_results(dataset, train_seq, train_label, decoded_seq, cluster_list):
	for i in range(len(train_seq)):
		print('label: {}, cluster: {}'.format(train_label[i], cluster_list[i]))

		if PLT:
			fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,15))
			fig.suptitle('upper: original sequence, lower: decoded sequence')
			ax[0].plot(train_seq[i])
			ax[1].plot(decoded_seq[i])
			plt.show()
			plt.close()

def inference(args):
	data = Dataset(args['dataset'])

	model = DeepTemporalClustering(params={
			'n_clusters': args['n_clusters'],
			'L': data.L,
			'K': data.K,
			'n_filters_CNN': 50,
			'kernel_size': 10,
			'P': 10,
			'n_filters_RNN_list': [50, 1],
			'alpha': 1.0
			})

	dec_ckpt_path = os.path.join('dtc_ckpt','model.ckpt')
	# dec_ckpt_path = os.path.join('ae_ckpt','model.ckpt')
	saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
	with tf.Session() as sess:
		saver.restore(sess, dec_ckpt_path)
		X_list, z_list, cluster_list = sess.run([model.auto_encoder.decoder, model.auto_encoder.encoder, model.pred], 
					feed_dict={model.auto_encoder.input_:data.train_seq,
							model.auto_encoder.input_batch_size:len(data.train_seq)})

	show_results(args['dataset'], data.train_seq, data.train_label, X_list, cluster_list)

def main():
	parser = configargparse.ArgParser()
	parser.add('-d', '--dataset' ,dest='dataset', type=str, default='synthetic', help='Name of Dataset')
	parser.add('-k', '--n_clusters', dest='n_clusters', type=int, default=4 ,help='Number of Clusters')
	args = vars(parser.parse_args())

	inference(args)

if __name__ == '__main__':
	main()