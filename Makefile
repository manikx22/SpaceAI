.PHONY: setup train run test

setup:
	python -m pip install -r requirements.txt

train:
	python preprocess.py
	python train_anomaly.py
	python train_lstm.py

run:
	streamlit run app.py --server.address 127.0.0.1 --server.port 8501

test:
	pytest -q
