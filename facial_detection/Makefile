format:
	black *.py
install:
	pip install -r requirements.txt
run: 
	python cnn.py
lint:
	pylint --disable=R,C *.py 

all: install format lint run
