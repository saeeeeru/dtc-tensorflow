import os, time

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_shape(name, tensor):
	print('shape of {} is {}'.format(name, tensor.shape))

class AutoEncoder(tf.keras.Model):
	"""AutoEncoder using Keras for TensorFlow 2.x compatibility"""
	def __init__(self, args):
		super(AutoEncoder, self).__init__()
		self.__dict__.update(args)

		# Encoder layers
		if self.K == 1:
			self.conv1d = tf.keras.layers.Conv1D(
				filters=self.n_filters_CNN,
				kernel_size=self.kernel_size,
				activation='relu',
				padding='same'
			)
			self.max_pool = tf.keras.layers.MaxPooling1D(
				pool_size=self.kernel_size,
				strides=self.P
			)
		else:
			self.conv2d_enc = tf.keras.layers.Conv2D(
				filters=self.n_filters_CNN,
				kernel_size=(self.kernel_size, self.K),
				strides=(self.P, self.P),
				padding='same'
			)

		# Bidirectional LSTM layers
		self.bi_lstm_layers = []
		for n_filters in self.n_filters_RNN_list:
			self.bi_lstm_layers.append(
				tf.keras.layers.Bidirectional(
					tf.keras.layers.LSTM(n_filters, return_sequences=True)
				)
			)

		# Decoder layers
		if self.K == 1:
			self.conv1d_dec = tf.keras.layers.Conv1D(
				filters=self.K,
				kernel_size=self.kernel_size,
				activation=None,
				padding='same'
			)
		else:
			self.conv2d_dec = tf.keras.layers.Conv2DTranspose(
				filters=1,
				kernel_size=(self.kernel_size, self.K),
				strides=(self.P, self.P),
				padding='same'
			)

	def encode(self, x):
		"""Encoder network"""
		print_shape('input', x)

		if self.K == 1:
			conv_out = self.conv1d(x)
			print_shape('conv out', conv_out)
			max_pooled = self.max_pool(conv_out)
			print_shape('max pooled', max_pooled)
		else:
			x_expanded = tf.expand_dims(x, axis=3)
			conv_out = self.conv2d_enc(x_expanded)
			print_shape('conv out', conv_out)
			max_pooled = tf.reshape(conv_out, shape=[tf.shape(x)[0], conv_out.shape[1], conv_out.shape[3]])
			print_shape('max pooled', max_pooled)

		encoder_out = max_pooled
		for bi_lstm in self.bi_lstm_layers:
			encoder_out = bi_lstm(encoder_out)
		print_shape('encoder', encoder_out)

		return encoder_out

	def decode(self, encoded, input_batch_size):
		"""Decoder network"""
		if self.K == 1:
			encoder_tmp = tf.expand_dims(encoded, axis=3)
			upsampled_tmp = tf.image.resize(encoder_tmp,
							size=[self.L + self.kernel_size - 1, 1])
			upsampled = tf.reshape(upsampled_tmp, shape=[-1, upsampled_tmp.shape[1], upsampled_tmp.shape[2]])
			print_shape('upsampled', upsampled)
			decoder = self.conv1d_dec(upsampled)
		else:
			encoder_tmp = tf.expand_dims(encoded, axis=2)
			print_shape('encoder tmp', encoder_tmp)
			decoder_tmp = self.conv2d_dec(encoder_tmp)
			# Resize to match original dimensions
			decoder_tmp = tf.image.resize(decoder_tmp, size=[self.L, self.K])
			decoder = tf.reshape(decoder_tmp, shape=[input_batch_size, self.L, self.K])

		print_shape('decoder', decoder)
		return decoder

	def call(self, x, input_batch_size):
		"""Forward pass"""
		encoded = self.encode(x)
		decoded = self.decode(encoded, input_batch_size)
		return encoded, decoded


class DeepTemporalClustering(object):
	"""Deep Temporal Clustering model for TensorFlow 2.x"""
	def __init__(self, params):
		self.__dict__ = params.copy()

		self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
		self.auto_encoder = AutoEncoder(self.__dict__)

		# Initialize mu as a trainable variable
		self.mu = None  # Will be initialized after first forward pass

	def initialize_mu(self, z_shape):
		"""Initialize cluster centers"""
		if self.mu is None:
			self.mu = tf.Variable(
				tf.zeros(shape=[self.n_clusters, z_shape[1], z_shape[2]]),
				name='mu',
				trainable=True
			)

	def soft_assignment(self, embeddings, cluster_centers):
		"""Implemented a soft assignment as the probability of assigning sample i to cluster j.

		Args:
			- embeddings: (N, L_tmp, dim)
			- cluster_centers: (n_clusters, L_tmp, dim)

		Return:
			- q_ij: (N, n_clusters)
		"""
		def _pairwise_euclidean_distance(a, b):
			return tf.norm(tf.expand_dims(a, axis=1) - b, 'euclidean', axis=(2, 3))

		dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
		q = 1.0 / (1.0 + dist**2 / self.alpha)**((self.alpha + 1.0) / 2.0)
		q = (q / tf.reduce_sum(q, axis=1, keepdims=True))

		return q

	def target_distribution(self, q):
		"""Calculate target distribution"""
		p = q**2 / q.sum(axis=0)
		p = p / p.sum(axis=1, keepdims=True)
		return p

	def kl_divergence(self, target, pred):
		"""Calculate KL divergence loss"""
		return tf.reduce_mean(tf.reduce_sum(target * tf.math.log(target / (pred)), axis=1))

	def get_assign_cluster_centers_op(self, features):
		"""Initialize cluster centers using KMeans"""
		print('Start training KMeans')
		kmeans = self.kmeans.fit(features.reshape(len(features), -1))
		print('Finish training KMeans')
		return kmeans.cluster_centers_.reshape(self.n_clusters, features.shape[1], features.shape[2])

	@tf.function
	def ae_train_step(self, x, input_batch_size, optimizer):
		"""Single training step for autoencoder"""
		with tf.GradientTape() as tape:
			encoded, decoded = self.auto_encoder(x, input_batch_size)
			loss = tf.reduce_mean(tf.square(x - decoded))

		gradients = tape.gradient(loss, self.auto_encoder.trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.auto_encoder.trainable_variables))
		return loss, encoded, decoded

	@tf.function
	def dtc_train_step(self, x, p, input_batch_size, optimizer):
		"""Single training step for DTC"""
		with tf.GradientTape() as tape:
			encoded, decoded = self.auto_encoder(x, input_batch_size)
			q = self.soft_assignment(encoded, self.mu)
			loss_kl = self.kl_divergence(p, q)
			loss_ae = tf.reduce_mean(tf.square(x - decoded))
			loss = loss_kl + 0.01 * loss_ae

		trainable_vars = self.auto_encoder.trainable_variables + [self.mu]
		gradients = tape.gradient(loss, trainable_vars)
		optimizer.apply_gradients(zip(gradients, trainable_vars))
		return loss, loss_kl, loss_ae, encoded, decoded, q

	def forward(self, x, input_batch_size):
		"""Forward pass without gradient computation"""
		encoded, decoded = self.auto_encoder(x, input_batch_size)
		if self.mu is not None:
			q = self.soft_assignment(encoded, self.mu)
			pred = tf.argmax(q, axis=1)
			return encoded, decoded, q, pred
		return encoded, decoded, None, None
