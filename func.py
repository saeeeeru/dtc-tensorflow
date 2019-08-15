import os

import scipy, configargparse
import sklearn.base
import bhtsne

import numpy as np
import matplotlib.pyplot as plt

cmap = plt.get_cmap("tab10") 

def flatten(data, depth=-1):
	return [element for item in data for element in (flatten(item, depth - 1) if depth != 0 and hasattr(item, '__iter__') else [item])]

def generate_args():
	parser = configargparse.ArgParser()
	parser.add('-d', '--dataset', dest='dataset', type=str, default='synthetic' ,help='Name of Dataset')
	parser.add('-k', '--n_clusters', dest='n_clusters', type=int, default=4 ,help='Number of Clusters')
	args = vars(parser.parse_args())

	return args

class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
	"""docstring for BHTSNE"""
	def __init__(self, args):
		self.__dict__ = args.copy()

	def fit_and_plot(self, X, label_list, cluster_list):
		data_tsne = bhtsne.tsne(
			X.astype(scipy.float64),
			dimensions=self.dimensions,
			perplexity=self.perplexity,
			theta=self.theta,
			rand_seed=self.rand_seed)


		xmin, xmax = data_tsne[:,0].min(), data_tsne[:,0].max()
		ymin, ymax = data_tsne[:,1].min(), data_tsne[:,1].max()

		# split each label
		data_dict = {str(label): np.array([data_tsne[idx] for idx, cluster in enumerate(cluster_list) if cluster == label]) for label in range(len(np.unique(label_list)))}

		plt.figure(figsize=(15,10))
		for m, data in data_dict.items():
			if data is not []:
				plt.scatter(data[:][0],data[:][1],cmap=cmap(int(m)),label=f'label.{m}: {len(data)}', alpha=0.5)
		plt.legend()
		# plt.axis([xmin,xmax,ymin,ymax])
		plt.xlabel('component 0')
		plt.ylabel('component 1')
		plt.title('t-SNE visualization')

		savedir = os.path.join('_fig','tsne')
		if not os.path.exists(savedir):
			os.mkdir(savedir)

		plt.savefig(os.path.join(savedir,'clustering_result.png'))
		plt.close()