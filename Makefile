.PHONY: install dev test lint

install:
	pip install -r requirements.txt

dev:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000 2>/dev/null || python main.py

test:
	python -m pytest tests/ -v

lint:
	python -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py')]; print('✅ All files pass syntax check')"
