
# --- Documentation Variables ---
DOCS_DIR = source
SPHINX_BUILD_DIR = $(CURDIR)/_build/html

# Python executable for Sphinx (assuming virtual environment in venv)
# Adjust if your Python setup is different
PYTHON = $(shell command -v python3 || command -v python)  
VENV_PYTHON = $(CURDIR)/../venv/bin/python

# Ensure the virtual environment's python is used if it exists
ifeq ($(wildcard $(VENV_PYTHON)),)
    # Virtual environment not found, fall back to system python
    PYTHON_EXECUTABLE = $(PYTHON)
else
    PYTHON_EXECUTABLE = $(VENV_PYTHON)
endif

# --- Documentation Targets ---

.PHONY: doc sphinx clean_doc

# Main documentation target: builds Doxygen, then Sphinx
doc: sphinx

# Sphinx target
# Depends on the Doxygen XML output to ensure it's up-to-date
$(SPHINX_BUILD_DIR)/index.html: $(wildcard $(DOCS_DIR)/*.rst $(DOCS_DIR)/**/*.rst $(DOCS_DIR)/*.py)
	@echo "Generating Sphinx HTML documentation..."
	mkdir -p $(@D) # Ensure the Sphinx build directory exists
	$(PYTHON_EXECUTABLE) -m sphinx -b html $(DOCS_DIR) $(SPHINX_BUILD_DIR)

sphinx: $(SPHINX_BUILD_DIR)/index.html

# Clean documentation build artifacts
clean_doc:
	@echo "Cleaning documentation build files..."
	rm -rf $(CURDIR)/_build
