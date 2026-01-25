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

	log_interval = int(finetune_iteration/100)

	# Phase 1: parameter(Auto-Encoder) finetuning
	if pretrained_ae_ckpt_path == None:
		ae_ckpt_path = os.path.join('ae_ckpt', 'model')

		print('Parameter(AE) finetuning')

		# Create optimizer with learning rate schedule
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=0.1,
			decay_steps=20000,
			decay_rate=0.1,
			staircase=True
		)
		ae_optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

		# Build model with sample input
		sample_input = tf.constant(data.train_seq[:batch_size], dtype=tf.float32)
		_ = model.auto_encoder(sample_input, batch_size)

		for _iter, (batch_seq, batch_label, _) in enumerate(data.gen_next_batch(batch_size=batch_size, iteration=finetune_iteration)):
			batch_seq = tf.constant(batch_seq, dtype=tf.float32)
			loss, _, _ = model.ae_train_step(batch_seq, batch_size, ae_optimizer)

			if _iter % log_interval == 0:
				print('[AE-finetune] iter: {}\tloss: {}'.format(_iter, loss.numpy()))

		# Save model weights
		os.makedirs(os.path.dirname(ae_ckpt_path), exist_ok=True)
		model.auto_encoder.save_weights(ae_ckpt_path)

	else:
		ae_ckpt_path = pretrained_ae_ckpt_path
		# Build model with sample input before loading weights
		sample_input = tf.constant(data.train_seq[:batch_size], dtype=tf.float32)
		_ = model.auto_encoder(sample_input, batch_size)
		model.auto_encoder.load_weights(ae_ckpt_path)

	# Phase 2: parameter optimization
	dtc_ckpt_path = os.path.join('dtc_ckpt', 'model')
	print('Parameter(DTC) optimization')

	# Load autoencoder weights if not already loaded
	if pretrained_ae_ckpt_path == None:
		# Weights already saved, reload them
		model.auto_encoder.load_weights(ae_ckpt_path)

	# Initialize mu
	train_seq_tensor = tf.constant(data.train_seq, dtype=tf.float32)
	z, _ = model.auto_encoder(train_seq_tensor, len(data.train_seq))
	z = z.numpy()

	model.initialize_mu(z.shape)
	cluster_centers = model.get_assign_cluster_centers_op(z)
	model.mu.assign(cluster_centers)

	# DTC optimizer
	dtc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

	log_interval = int(optimization_iteration/10)
	for cur_epoch in range(optimization_iteration):
		# Calculate q and p
		train_seq_tensor = tf.constant(data.train_seq, dtype=tf.float32)
		z, _ = model.auto_encoder(train_seq_tensor, len(data.train_seq))
		q = model.soft_assignment(z, model.mu).numpy()
		p = model.target_distribution(q)

		# per one epoch
		loss_train_list, loss_kl_train_list, loss_seq_train_list = [], [], []
		for _iter, (batch_seq, batch_label, batch_idxs) in enumerate(data.gen_next_batch(batch_size=batch_size, epoch=1)):
			batch_p = tf.constant(p[batch_idxs], dtype=tf.float32)
			batch_seq = tf.constant(batch_seq, dtype=tf.float32)

			loss, loss_kl, loss_seq, encoded, decoded, pred_q = model.dtc_train_step(
				batch_seq, batch_p, len(batch_seq), dtc_optimizer
			)

			loss_train_list.append(loss.numpy())
			loss_kl_train_list.append(loss_kl.numpy())
			loss_seq_train_list.append(loss_seq.numpy())

		# validation
		if cur_epoch % 10 == 0:
			test_seq_tensor = tf.constant(data.test_seq, dtype=tf.float32)
			z_val, decoded_val = model.auto_encoder(test_seq_tensor, len(data.test_seq))
			q_val = model.soft_assignment(z_val, model.mu).numpy()
			p_val = model.target_distribution(q_val)

			loss_kl_val = model.kl_divergence(
				tf.constant(p_val, dtype=tf.float32),
				tf.constant(q_val, dtype=tf.float32)
			).numpy()
			loss_seq_val = tf.reduce_mean(tf.square(test_seq_tensor - decoded_val)).numpy()
			loss_val = loss_kl_val + 0.01 * loss_seq_val

			print_result(cur_epoch, optimization_iteration,
				loss_train_list[-1], loss_seq_train_list[-1], loss_kl_train_list[-1],
				loss_val, loss_seq_val, loss_kl_val)

	# Save DTC model
	os.makedirs(os.path.dirname(dtc_ckpt_path), exist_ok=True)
	model.auto_encoder.save_weights(dtc_ckpt_path)
	# Save mu separately
	np.save(os.path.join('dtc_ckpt', 'mu.npy'), model.mu.numpy())

def main():
	args = generate_args()

	train(args)

if __name__ == '__main__':
	main()
