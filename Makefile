.PHONY: install-dev test

install-dev:
	@echo "Installing development dependencies..."
	@pip install -r requirements-dev.txt

test:
	@echo "Running test suite..."
	@pytest