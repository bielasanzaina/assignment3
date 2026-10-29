import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from nb.nb import NaiveBayes


# The test documents are the "worked example" documents from Jurafsky and
# Martin, Appendix B. The two classes are negative (0) and positive (1).
# With add-1 smoothing (Lidstone alpha=1), the textbook gives:
#   P(-) = 3/5,  P(+) = 2/5
#   P("predictable"|-) = 2/34   P("predictable"|+) = 1/29
#   P("no"|-)          = 2/34   P("no"|+)          = 1/29
#   P("fun"|-)         = 1/34   P("fun"|+)         = 2/29
LABEL_TO_ID_MAP = {
    "neg": 0,
    "pos": 1
}


def load_test_docs():
    """
    Reads the Jurafsky and Martin worked-example documents from
    tests/data/docs.txt into a DataFrame with the same columns used
    by the NaiveBayes class.

    :return: a pandas DataFrame with "author" and "text" columns
    """

    docs_path = Path(__file__).parent / "data" / "docs.txt"

    rows = []

    with open(docs_path) as fp:

        for line in fp:

            line = line.strip()

            if not line:
                continue

            label, text = line.split("\t")

            rows.append({
                "author": LABEL_TO_ID_MAP[label],
                "text": text
            })

    return pd.DataFrame(rows)


def train_model():
    """
    Trains the NaiveBayes class on the worked-example documents with
    add-1 smoothing, so results are directly comparable to the textbook.

    :return: a trained NaiveBayes model
    """

    df = load_test_docs()

    model = NaiveBayes(alpha=1.0)

    model.train(df)

    return model


def test_nb_class():
    """
    End-to-end check of the NaiveBayes class: training on the textbook
    documents and testing on the textbook test document ("predictable
    with no fun") should predict the negative class, as in the book.
    """

    model = train_model()

    test_df = pd.DataFrame([{
        "text": "predictable with no fun"
    }])

    predictions = model.test(test_df)

    assert predictions == [LABEL_TO_ID_MAP["neg"]]


def test_priors():
    """
    The priors should match the textbook: P(-) = 3/5 and P(+) = 2/5.
    """

    model = train_model()

    assert model.priors[LABEL_TO_ID_MAP["neg"]] == pytest.approx(3 / 5)
    assert model.priors[LABEL_TO_ID_MAP["pos"]] == pytest.approx(2 / 5)


def test_likelihoods():
    """
    The likelihoods should match the textbook values computed with
    add-1 smoothing.
    """

    model = train_model()

    neg = LABEL_TO_ID_MAP["neg"]
    pos = LABEL_TO_ID_MAP["pos"]

    expected = {
        "predictable": (2 / 34, 1 / 29),
        "no": (2 / 34, 1 / 29),
        "fun": (1 / 34, 2 / 29)
    }

    for word, (p_neg, p_pos) in expected.items():

        w = model.vocabulary[word]

        assert model.likelihoods[neg, w] == pytest.approx(p_neg)
        assert model.likelihoods[pos, w] == pytest.approx(p_pos)


def test_likelihoods_sum_to_one():
    """
    Along the axis for a given class, every element is a probability
    estimate for a word in that class, so the probabilities over the
    whole vocabulary must sum to 1 for each class.
    """

    model = train_model()

    sums = np.sum(model.likelihoods, axis=1)

    for class_sum in sums:
        assert class_sum == pytest.approx(1.0)
