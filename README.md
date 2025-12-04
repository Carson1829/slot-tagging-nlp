README: Slot Tagging
============================================================

------------------------------------------------------------
1. Overview
------------------------------------------------------------


This repository contains code for training and evaluating a sequence-tagging model using PyTorch. It includes utilities for data loading, tokenization, vocabulary construction, hyperparameter tuning, and preparing trimmed GloVe embeddings.

The workflow includes:

- A main interface script (main.py)

- A GloVe preprocessing script (glove.py)

- A hyperparameter search script (tuning.py)

- Model architecture and training utilities (model.py, train.py, data.py)

------------------------------------------------------------
2. Running the main program
------------------------------------------------------------


Command: python main.py train.csv test.csv test_pred.csv


This command should:

- Train your best model using the data in train.csv

- Load and preprocess samples in test.csv

- Generate predictions using the trained model

- Write those predictions to test_pred.csv



Make sure your main.py handles:

- Loading embedding.pt (if you use trimmed GloVe)

- Constructing the correct model architecture

- Restoring your best hyperparameters

- Saving predictions in the correct output format

------------------------------------------------------------
3. Environemt setup
------------------------------------------------------------

Install dependencies:

pip install .


------------------------------------------------------------
4. Preparing Trimmed GloVe Embeddings
------------------------------------------------------------

Before training, generate the trimmed GloVe matrix that matches your vocabulary:

Command: python glove.py


This produces:

embedding.pt


The training code will load this instead of parsing the full GloVe text file.


------------------------------------------------------------
5. Running Hyperparameter Tuning
------------------------------------------------------------

To search for the best configuration:

Command: python tuning.py

This script:

- Tests multiple configurations

- Logs performance

- Outputs the best hyperparameters, which should then be used inside main.py