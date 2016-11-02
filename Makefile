install:
	pip install -r requirements.txt
	touch local/inp.txt

download:
	python -m nltk.downloader punkt
	python -m nltk.downloader averaged_perceptron_tagger
	python -m nltk.downloader sentiwordnet
	python -m nltk.downloader wordnet

test:
	python -m project.tests.test_installation

word2vec.init:
	mkdir local/word2vec
	echo 0 > local/word2vec/count.txt
	touch local/word2vec/reviews.vocab

word2vec.build_vocab:
	python -m project.word2vec.build_vocab

word2vec.train:
	python -m project.word2vec.train
