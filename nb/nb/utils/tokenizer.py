class Tokenizer:
    """
    Interface for tokenizers. Every tokenizer turns a string of text into
    a list of token strings via a tokenize() method; concrete subclasses
    decide how the text is split.
    """

    def tokenize(self, text):
        """
        Splits the input text into a list of tokens.

        :param text: a string of raw text
        :return: a list of token strings
        """

        raise NotImplementedError


class WhitespaceTokenizer(Tokenizer):
    """
    Tokenizes text by splitting on whitespace, exactly as we did in
    Assignment 2.
    """

    def tokenize(self, text):
        """
        Splits the input text along whitespace.

        :param text: a string of raw text
        :return: a list of token strings
        """

        return text.split()
