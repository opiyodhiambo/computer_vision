format:
	black *.py
install:
	pip install -r requirements.txt
run: 
	python facial_detect.py
lint:
	pylint --disable=R,C,W0212 *.py 
build: 
	docker build .
all: install format lint run
