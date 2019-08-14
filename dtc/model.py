import os, time

import numpy
import tensorflow as tf
from sklearn.cluster import KMeans

layers = tf.contrib.layers
rnn = tf.contrib.rnn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def print_shape(name, tensor):
	print('shape of {} is {}'.format(name,tensor.shape))

class AutoEncoder(object):
	"""docstring for AutoEncoder"""
	def __init__(self, args):
		self.__dict__ = args.copy()
		self.input_ = tf.placeholder(tf.float32, shape=[None, self.L, self.K])
		self.input_batch_size = tf.placeholder(tf.int32, shape=[])

		self.layers = []

		with tf.name_scope('encoder'):
			self.encoder = self._encoder_network()

		with tf.name_scope('decoder'):
			self.decoder = self._decoder_network()

		with tf.name_scope('ae-train'):
			self.loss = tf.losses.mean_squared_error(self.input_, self.decoder)
			learning_rate = tf.train.exponential_decay(learning_rate=0.1, 
												global_step=tf.train.get_or_create_global_step(),
												decay_steps=20000,
												decay_rate=0.1,
												staircase=True)
			self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)


	def _encoder_network(self):
		print_shape('input', self.input_)
		used = tf.sign(tf.reduce_max(tf.abs(self.input_), reduction_indices=2))
		self.length = tf.cast(tf.reduce_sum(used, reduction_indices=1), tf.int32)

		if self.K == 1:
			conv_out = layers.convolution1d(inputs=self.input_,
										num_outputs=self.n_filters_CNN,
										kernel_size=self.kernel_size,
										activation_fn=tf.nn.leaky_relu)
			print_shape('conv out', conv_out)
			max_pooled = tf.layers.max_pooling1d(inputs=conv_out,
										pool_size=self.kernel_size,
										strides=self.P)
			print_shape('max pooled', max_pooled)
		else:
			W_conv_enc = tf.get_variable('W_conv_enc', shape=[self.kernel_size, self.K, 1, self.n_filters_CNN])
			conv_out = tf.nn.conv2d(input=tf.expand_dims(self.input_, axis=3),
								filter=W_conv_enc,
								strides=[1, self.P, self.P, 1],
								padding='SAME')
			print_shape('conv out', conv_out)
			max_pooled = tf.reshape(conv_out, shape=[self.input_batch_size,conv_out.shape[1],conv_out.shape[3]])
			print_shape('max pooled', max_pooled)

		cell_fw_list = [rnn.LSTMCell(n_filters_RNN) for n_filters_RNN in self.n_filters_RNN_list]
		cell_bw_list = [rnn.LSTMCell(n_filters_RNN) for n_filters_RNN in self.n_filters_RNN_list]

		encoder, encoder_state_fw, encoder_state_bw = rnn.stack_bidirectional_dynamic_rnn(
														cells_fw=cell_fw_list, cells_bw=cell_bw_list, inputs=max_pooled, 
														# sequence_length=self.length,
														dtype=tf.float32, time_major=False, scope=None)
		print_shape('encoder', encoder)

		return encoder

	def _decoder_network(self):
		if self.K == 1:
			encoder_tmp = tf.expand_dims(self.encoder, axis=3)
			
			upsampled_tmp = tf.image.resize_images(encoder_tmp, 
							size=[self.encoder.shape[1]*self.P,1])
							# method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			upsampled = tf.reshape(upsampled_tmp, shape=[-1,upsampled_tmp.shape[1],upsampled_tmp.shape[2]])
			print_shape('upsampled', upsampled)

			decoder = layers.convolution1d(inputs=upsampled,
										num_outputs=self.K,
										kernel_size=self.kernel_size,
										activation_fn=None)
		else:
			encoder_tmp = tf.expand_dims(self.encoder, axis=2)
			print_shape('encoder tmp', encoder_tmp)
			W_conv_dec = tf.get_variable('W_conv_dec', shape=[self.kernel_size, self.K, 1, encoder_tmp.shape[3]])
			decoder_tmp = tf.nn.conv2d_transpose(value=encoder_tmp,
							filter=W_conv_dec,
							output_shape=[self.input_batch_size, self.L, self.K, 1],
							strides=[1, self.P, self.P, 1],
							padding='SAME')
			decoder = tf.reshape(decoder_tmp, shape=[self.input_batch_size, self.L, self.K])

		print_shape('decoder', decoder)

		return decoder

class DeepTemporalClustering(object):
	"""docstring for DeepTemporalClustering"""
	def __init__(self, params):
		self.__dict__ = params.copy()

		self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
		self.auto_encoder = AutoEncoder(self.__dict__)

		self.z = self.auto_encoder.encoder
		self.y = self.auto_encoder.decoder

		z_shape = self.z.shape
		self.mu = tf.Variable(tf.zeros(shape=[self.n_clusters,z_shape[1],z_shape[2]]), name='mu')
		# self.mu = tf.Variable(tf.zeros(shape=[self.n_clusters,z_shape[1]]), name='mu')

		with tf.name_scope('distribution'):
			self.q = self._soft_assignment(self.z, self.mu)
			self.p = tf.placeholder(tf.float32, shape=(None, self.n_clusters))

			self.pred = tf.argmax(self.q, axis=1)

		with tf.name_scope('dtc-train'):
			# self.loss = self._kl_divergence(self.p, self.q)
			self.loss_kl = self._kl_divergence(self.p, self.q) 
			self.loss = self.loss_kl + 0.01 * self.auto_encoder.loss
			self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

	def _soft_assignment(self, embeddings, cluster_centers):
		"""Implemented a soft assignment as the  probability of assigning sample i to cluster j.

		Args:
			- embeddings: (N, L_tmp, dim)
			- cluster_centers: (n_clusters, L_tmp, dim)

		Return:
			- q_ij: (N, n_clusters)
		"""
		def _pairwise_euclidean_distance(a, b):
			return tf.norm(tf.expand_dims(a, axis=1) - b, 'euclidean', axis=(2,3))

		dist = _pairwise_euclidean_distance(embeddings, cluster_centers)
		q = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+1.0)/2.0)
		q = (q/tf.reduce_sum(q, axis=1, keepdims=True))
		
		return q

	def target_distribution(self, q):
		p = q**2 / q.sum(axis=0)
		p = p / p.sum(axis=1, keepdims=True)
		return p

	def _kl_divergence(self, target, pred):
		return tf.reduce_mean(tf.reduce_sum(target*tf.log(target/(pred)), axis=1))

	def get_assign_cluster_centers_op(self, features):
		# init mu
		print('Start training KMeans')
		kmeans = self.kmeans.fit(features.reshape(len(features),-1))
		print('Finish training KMeans')
		return tf.assign(self.mu, kmeans.cluster_centers_.reshape(self.n_clusters,features.shape[1],features.shape[2]))
