install:
	pip3 install -r requirements.txt
	touch local/inp.txt

download:
	python3 -m nltk.downloader punkt
	python3 -m nltk.downloader averaged_perceptron_tagger
	python3 -m nltk.downloader sentiwordnet
	python3 -m nltk.downloader wordnet

test:
	python3 -m project.tests.test_installation
