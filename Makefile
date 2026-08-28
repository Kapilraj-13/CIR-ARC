.PHONY: setup generate eval baseline test leakage clean

PYTHON = .venv/Scripts/python.exe
PYTHONPATH_SET = set PYTHONPATH=. &&

setup:
	pip install -r requirements.txt

generate:
	$(PYTHONPATH_SET) $(PYTHON) scripts/generate_data.py --n_per_rule 1000 --n_per_pair 500

eval:
	$(PYTHONPATH_SET) $(PYTHON) scripts/run_benchmark.py

baseline:
	$(PYTHONPATH_SET) $(PYTHON) scripts/run_benchmark.py

test:
	$(PYTHON) -m pytest tests/ -v --cov=core --cov=dsl --cov=generators --cov=eval --cov=baselines

leakage:
	$(PYTHONPATH_SET) $(PYTHON) scripts/check_leakage.py

clean:
	if exist data\synthetic\train rd /s /q data\synthetic\train
	if exist data\synthetic\held_out rd /s /q data\synthetic\held_out
