# Machine Learning From Scratch: Linear Regression
This repo is for implementing **linear regression** on the famous Red Wine Quality dataset.

Author: Kisaragi.Z (*An enthusiast for sharing knowledge.*

## Download Data
download this data and put it under `data/`
-  https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009?resource=download

## Environment Preparation
### poetry 
- https://python-poetry.org/docs/#installing-with-the-official-installer
- simple downloading instruction on Mac OS
    ```bash 
    # download poetry
    curl -sSL https://install.python-poetry.org | python3 -

    # add poetry path to ~/.zshrc 
    export PATH="$HOME/.local/bin:$PATH >> ~/.zshrc
    ```

## Project Structure
- data
- module
	- dataset.py
	- trainer.py
	- metric.py
- notebooks
	- 0-data-preparation
		- 01-visualiza-data.ipynb
		- 02-check-dataset.ipynb
	- 1-implement-training-testing
		- train.py
		- param.py
	- 2-evaluate-results
		- 01-check-metric.ipynb
		- 02-evaluate-training-testing.ipynb
- poetry.lock
- pyproject.toml