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
