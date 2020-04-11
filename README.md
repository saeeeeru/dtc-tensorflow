# dtc-tensorlow
This repository is not official source code of [「Deep Temporal Clustering: Fully Unsupervised Learning of Time-Domain Features」](https://arxiv.org/abs/1802.01059).  
We use synthetic dataset (a superposition of sinusoids).  
please let me know if you have any questions to email: saeeeeru29@gmail.com

## Usage
## requirements
- Written for Python3.x
- Please check required Python Library in ./requirements.txt

## Installation
1. Install required Python Library

~~~
pip install -r ./requirements.txt
~~~

### Run demo
1. Learn Deep Temporal Clustering Model

	```
	sh demo.sh
	```

2. Inference clustering results of learnd Model

	```
	python inference.py
	```
