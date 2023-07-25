format:
	black *.py
install:
	pip install -r requirements.txt
run: 
	python facial-detect.py
lint:
	pylint --disable=R,C *.py 

all: install format lint run
