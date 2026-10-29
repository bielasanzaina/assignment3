import numpy as np

from nb.utils.tokenizer import WhitespaceTokenizer


class NaiveBayes:
    """
    A multinomial Naive Bayes classifier with Lidstone smoothing.

    The classifier is trained on a pandas DataFrame with a "text" column
    and an "author" column holding integer class labels. Training computes
    the class priors and the word likelihoods; testing scores unseen
    documents in log space and returns the most probable class for each.
    """

    def __init__(self, alpha=0.1, tokenizer=None):
        """
        Initializes the classifier.

        :param alpha: Lidstone smoothing parameter
        :param tokenizer: object with a tokenize() method; defaults to
            whitespace tokenization, as in Assignment 2
        """

        self.alpha = alpha
        self.tokenizer = tokenizer if tokenizer else WhitespaceTokenizer()

        # Set during training.
        self.vocabulary = {}
        self.priors = None
        self.likelihoods = None

    def train(self, df):
        """
        Fits the model by computing priors and likelihoods from the
        training DataFrame.

        :param df: a pandas DataFrame with "author" and "text" columns
        :return: self, so calls can be chained
        """

        # Build the vocabulary from the training texts.
        self.vocabulary = {}

        for text in df["text"]:

            for token in self.tokenizer.tokenize(text):

                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)

        n_docs = df.shape[0]
        n_classes = df["author"].nunique()

        # Compute prior probabilities.
        self.priors = np.zeros(shape=n_classes)

        for c in range(n_classes):

            self.priors[c] = (
                np.sum(df["author"] == c) / n_docs
            )

        # Create the bag-of-words training matrix.
        training_matrix = np.zeros(
            shape=(n_docs, len(self.vocabulary))
        )

        for doc_idx, text in enumerate(df["text"]):

            for token in self.tokenizer.tokenize(text):

                training_matrix[
                    doc_idx,
                    self.vocabulary[token]
                ] += 1

        # Count words for each class.
        word_counts_per_class = np.zeros(
            shape=(n_classes, len(self.vocabulary))
        )

        for c in range(n_classes):

            class_docs = training_matrix[
                df["author"] == c
            ]

            word_counts_per_class[c] = np.sum(
                class_docs,
                axis=0
            )

        # Compute likelihoods using Lidstone smoothing. The matrix has
        # shape (n_classes, |V|), and the denominator adds alpha once per
        # vocabulary entry, so each row forms a proper probability
        # distribution over the vocabulary (each row sums to 1). This is
        # verified by a unit test in tests/test_naive_bayes.py.
        self.likelihoods = np.zeros(
            shape=(n_classes, len(self.vocabulary))
        )

        for c in range(n_classes):

            denominator = (
                np.sum(word_counts_per_class[c])
                + self.alpha * len(self.vocabulary)
            )

            for w in range(len(self.vocabulary)):

                self.likelihoods[c, w] = (
                    word_counts_per_class[c, w] + self.alpha
                ) / denominator

        return self

    def test(self, df):
        """
        Predicts a class label for each document in the test DataFrame.

        Unknown words (words never seen in training) are skipped, as in
        the textbook algorithm.

        :param df: a pandas DataFrame with a "text" column
        :return: a list of predicted class labels
        """

        if self.priors is None or self.likelihoods is None:
            raise RuntimeError(
                "The model must be trained before calling test()."
            )

        class_predictions = []

        for text in df["text"]:

            test_vector = np.zeros(
                shape=len(self.vocabulary)
            )

            for word in self.tokenizer.tokenize(text):

                index = self.vocabulary.get(word)

                if index is not None:
                    test_vector[index] += 1

            preds = np.log(self.priors) + np.sum(
                test_vector * np.log(self.likelihoods),
                axis=1
            )

            yhat = np.argmax(preds)

            class_predictions.append(yhat)

        return class_predictions
