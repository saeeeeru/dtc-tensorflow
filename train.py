import os

import configargparse
import numpy as np
import tensorflow as tf

from dataset import Dataset
from dtc.model import DeepTemporalClustering

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(args, \
			batch_size=8, \
			initialize_iteration=100, \
			finetune_iteration=100, \
			pretrained_ae_ckpt_path=None):
	dataset = args['dataset']

	data = Dataset(dataset)

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

	saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
	log_interval = int(initialize_iteration/100)

	# Phase 1: parameter(Auto-Encoder) initialization
	if pretrained_ae_ckpt_path == None:
		ae_ckpt_path = os.path.join('ae_ckpt','model.ckpt')

		print('Parameter(AE) initialization')
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for _iter, (batch_seq, batch_label, _) in enumerate(data.gen_next_batch(batch_size=batch_size,iteration=finetune_iteration)):
				_, loss = sess.run([model.auto_encoder.optimizer, model.auto_encoder.loss], \
								feed_dict={model.auto_encoder.input_: batch_seq})

				if _iter % log_interval == 0:
					print('[AE-finetune] iter: {}\tloss: {}'.format(_iter,loss))

			saver.save(sess, ae_ckpt_path)

	else:
		ae_ckpt_path = pretrained_ae_ckpt_path

	# Phase 2: parameter optimization
	dec_ckpt_path = os.path.join('dtc_ckpt','model.ckpt')
	print('Parameter(DTC) initialization')
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# save.restore(sess, ae_ckpt_path)

		# init mu
		length_list, z = sess.run([model.auto_encoder.length, model.auto_encoder.encoder], feed_dict={model.auto_encoder.input_: data.train_seq})
		assign_mu_op = model.get_assign_cluster_centers_op(z)
		_ = sess.run(assign_mu_op)

		for cur_epoch in range(50):
			q = sess.run(model.q, 
				feed_dict={model.auto_encoder.input_: data.train_seq, \
					model.auto_encoder.input_batch_size: len(data.train_seq)})
			p = model.target_distribution(q)

			# per one epoch
			for _iter, (batch_seq, batch_label, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size, epoch=1)):
				
				batch_p = p[batch_idxs]
				_, loss, pred, decoder = sess.run([model.optimizer, model.loss, model.pred, model.y], \
									feed_dict={model.auto_encoder.input_: batch_seq, \
											model.auto_encoder.input_batch_size: len(batch_seq), \
											model.p: batch_p})

			print('[DTC] epoch: {}\tloss: {}\tacc: {}'.format(cur_epoch, loss, pred))
			saver.save(sess, dec_ckpt_path)

def main():
	parser = configargparse.ArgParser()
	parser.add('-d', '--dataset', dest='dataset', type=str, default='synthetic' ,help='Name of Dataset')
	parser.add('-k', '--n_clusters', dest='n_clusters', type=int, default=4 ,help='Number of Clusters')
	args = vars(parser.parse_args())

	train(args)

if __name__ == '__main__':
	main()