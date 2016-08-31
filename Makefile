install:
	pip3 install -r requirements.txt
	touch local/inp.txt

download:
	python3 -m nltk.downloader all

test:
	python3 -m project.tests.test_installation
