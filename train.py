import os

import numpy as np
import tensorflow as tf

from func import generate_args
from dataset import Dataset
from dtc.model import DeepTemporalClustering

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_result(cur, total, loss_all_train, loss_seq_train, loss_train, loss_all_val, loss_seq_val, loss_val):
	print('{0:d} / {1:d}\t train ({2:5.3f}, {3:5.3f}, {4:5.3f})\t val({5:5.3f}, {6:5.3f}, {7:5.3f})\t in order (total, seq, lat)' \
							.format(cur, total, loss_all_train, loss_seq_train, loss_train, loss_all_val, loss_seq_val, loss_val))

def train(args, \
			batch_size=8, \
			finetune_iteration=100, \
			optimization_iteration=100, \
			pretrained_ae_ckpt_path=None):
	dataset = args['dataset']

	data = Dataset(dataset)

	model = DeepTemporalClustering(params={
		'n_clusters': args['n_clusters'],
		'L': data.L,
		'K': data.K,
		'n_filters_CNN': 100,
		'kernel_size': 10,
		'P': 10,
		'n_filters_RNN_list': [50, 50],
		'alpha': 1.0
		})

	saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=None)
	log_interval = int(finetune_iteration/100)

	# Phase 1: parameter(Auto-Encoder) finetuning
	if pretrained_ae_ckpt_path == None:
		ae_ckpt_path = os.path.join('ae_ckpt','model.ckpt')

		print('Parameter(AE) finetuning')
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for _iter, (batch_seq, batch_label, _) in enumerate(data.gen_next_batch(batch_size=batch_size,iteration=finetune_iteration)):
				_, loss = sess.run([model.auto_encoder.optimizer, model.auto_encoder.loss], \
								feed_dict={model.auto_encoder.input_: batch_seq, model.auto_encoder.input_batch_size: batch_size})

				if _iter % log_interval == 0:
					print('[AE-finetune] iter: {}\tloss: {}'.format(_iter, loss))

			saver.save(sess, ae_ckpt_path)

	else:
		ae_ckpt_path = pretrained_ae_ckpt_path

	# Phase 2: parameter optimization
	dec_ckpt_path = os.path.join('dtc_ckpt','model.ckpt')
	print('Parameter(DTC) optimization')
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# save.restore(sess, ae_ckpt_path)

		# init mu
		z = sess.run(model.z, feed_dict={model.auto_encoder.input_: data.train_seq, model.auto_encoder.input_batch_size: len(data.train_seq)})
		assign_mu_op = model.get_assign_cluster_centers_op(z)
		
		_ = sess.run(assign_mu_op)

		log_interval = int(optimization_iteration/10)
		for cur_epoch in range(optimization_iteration):
			q = sess.run(model.q, feed_dict={model.auto_encoder.input_: data.train_seq, model.auto_encoder.input_batch_size: len(data.train_seq)})
			p = model.target_distribution(q)

			# per one epoch
			loss_train_list, loss_kl_train_list, loss_seq_train_list = [], [], []
			for _iter, (batch_seq, batch_label, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size, epoch=1)):
				batch_p = p[batch_idxs]
				_, loss, loss_kl, loss_seq, pred, decoder = sess.run([model.optimizer, model.loss, model.loss_kl, model.auto_encoder.loss, model.pred, model.y], \
																	feed_dict={model.auto_encoder.input_: batch_seq, \
																			model.auto_encoder.input_batch_size: len(batch_seq), \
																			model.p: batch_p})
				loss_train_list.append(loss); loss_kl_train_list.append(loss_kl); loss_seq_train_list.append(loss_seq)

			# validation
			if cur_epoch % 10 == 0:
				q = sess.run(model.q, feed_dict={model.auto_encoder.input_: data.test_seq, model.auto_encoder.input_batch_size: len(data.test_seq)})
				p = model.target_distribution(q)

				loss_val, loss_kl_val, loss_seq_val = sess.run([model.loss, model.loss_kl, model.auto_encoder.loss], \
															feed_dict={model.auto_encoder.input_: data.test_seq, \
																	model.auto_encoder.input_batch_size: len(data.test_seq), \
																	model.p: p})

				print_result(cur_epoch, optimization_iteration, \
					# np.mean(loss_train_list), np.mean(loss_seq_train_list), np.mean(loss_kl_train_list), \
					loss, loss_seq, loss_kl,\
					loss_val, loss_seq_val, loss_kl_val)

		saver.save(sess, dec_ckpt_path)

def main():
	args = generate_args()

	train(args)
	# train(args, ps.path.join('ar_ckpt','model.ckpt'))

if __name__ == '__main__':
	main()