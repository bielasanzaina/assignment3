# Naive Bayes Authorship Identification

This project refactors the Naive Bayes classifier from Assignment 2 into a
proper Python project. The classifier performs authorship identification
over speeches by President John F. Kennedy and President Lyndon B. Johnson:
given the labeled speeches in `data/kennedy` and `data/johnson`, it predicts
which president authored each of the unlabeled speeches in `data/unlabeled`.

The model is a multinomial Naive Bayes classifier with Lidstone smoothing,
implemented from scratch with numpy. Training computes the class priors and
the per-class word likelihoods; testing scores unseen documents in log space
and predicts the most probable author.

## Project layout

```
nb/
    data/               raw speech text (not tracked by git)
    nb/
        nb.py           the NaiveBayes class (train and test methods)
        bin/main.py     the command-line interface
        utils/          data loading and tokenization utilities
    tests/              pytest unit tests + the small test corpus they use
    pytest.ini          tells pytest where the tests live
    setup.py            installs the project as a package
    environment.yml     the conda environment for the class
```

## Installation

Create and activate the class conda environment, then install the package
in editable mode from this directory (the folder containing `setup.py`):

```
conda env create -f environment.yml
conda activate dsan5400
pip install -e .
```

## Usage

Run the command-line interface from this directory, pointing it at the data
folder:

```
python -m nb.bin.main -f data
```

This builds the training and test DataFrames, trains the `NaiveBayes`
class, prints the priors and the shape of the likelihood matrix, and prints
the predicted author for every unlabeled speech. The Lidstone smoothing
parameter can be changed with `-a` (default 0.1), e.g.
`python -m nb.bin.main -f data -a 1.0`.

The class can also be used directly:

```python
from nb.nb import NaiveBayes
from nb.utils.load_data import build_dataframe

training_df, test_df = build_dataframe("data")

model = NaiveBayes(alpha=0.1)
model.train(training_df)
predictions = model.test(test_df)
```

## Tests

The unit tests train the classifier on the worked-example documents from
Jurafsky and Martin (Appendix B) and check the priors, the likelihoods, and
that the likelihoods for each class sum to 1. Run them from this directory
with:

```
pytest tests/
```
