.PHONY: run install
.venv/bin/python:
	python3 -m venv .venv
install: .venv/bin/python
	.venv/bin/python -m pip install -r requirements.txt
run: install
	.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1
