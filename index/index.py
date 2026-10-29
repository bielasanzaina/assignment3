import argparse
import string
import xml.etree.ElementTree as ET

from collections import Counter

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer


class Tokenizer:
    """
    Interface for tokenizers. The universal properties shared by every
    tokenizer are (1) a tokenize() method that turns a string of text
    into a list of token strings, and (2) a normalize() step that
    lowercases tokens and strips punctuation from their edges. Concrete
    subclasses decide how the text itself gets split.
    """

    def tokenize(self, text):
        """
        Splits the input text into a list of tokens.

        :param text: a string of raw text
        :return: a list of token strings
        """

        raise NotImplementedError

    def normalize(self, tokens):
        """
        Performs basic normalization on tokenized output by lowercasing
        and removing punctuation. Tokens that are empty after
        normalization are dropped.

        :param tokens: a list of token strings
        :return: a list of normalized token strings
        """

        normalized = []

        for token in tokens:

            token = token.lower().strip(string.punctuation)

            if token:
                normalized.append(token)

        return normalized

    def __call__(self, text):
        """
        Tokenizes and then normalizes the input text.

        :param text: a string of raw text
        :return: a list of normalized token strings
        """

        return self.normalize(
            self.tokenize(text)
        )


class WhitespaceTokenizer(Tokenizer):
    """
    Tokenizes text by simply splitting on whitespace.
    """

    def tokenize(self, text):
        return text.split()


class OffTheShelfTokenizer(Tokenizer):
    """
    Tokenizes text with an off-the-shelf tokenizer, the Treebank word
    tokenizer from nltk.
    """

    def __init__(self):
        self._tokenizer = TreebankWordTokenizer()

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)


class NgramTokenizer(Tokenizer):
    """
    Tokenizes text into character n-grams, where n is a parameter of the
    tokenizer (default n=4). The n-grams are taken over characters, not
    words.
    """

    def __init__(self, n=4):
        self.n = n

    def tokenize(self, text):

        if len(text) < self.n:
            return [text]

        return [
            text[i:i + self.n]
            for i in range(len(text) - self.n + 1)
        ]


TOKENIZER_MAP = {
    "whitespace": WhitespaceTokenizer,
    "off-the-shelf": OffTheShelfTokenizer,
    "ngram": NgramTokenizer
}


def parse_collection(filepath):
    """
    Parses the XML collection and pulls out (docid, text) pairs.

    :param filepath: path to the XML file holding the collection
    :return: a list of (docid, text) tuples
    """

    tree = ET.parse(filepath)
    root = tree.getroot()

    documents = []

    for doc in root.findall("doc"):

        docid = doc.find("docid").text.strip()
        msgtext = doc.find("msgtext").text

        documents.append((docid, msgtext))

    return documents


def stem(tokens):
    """
    Stems tokens using the Porter Stemmer from nltk.

    :param tokens: a list of token strings
    :return: a list of stemmed token strings
    """

    stemmer = PorterStemmer()

    return [stemmer.stem(token) for token in tokens]


def build_index(documents, tokenizer, use_stemming):
    """
    Builds an inverted index over the collection. Each postings list
    contains (docid, term frequency) pairs.

    :param documents: a list of (docid, text) tuples
    :param tokenizer: a Tokenizer used to split document text
    :param use_stemming: whether to stem tokens before indexing
    :return: a dict mapping token -> list of [docid, tf] pairs
    """

    index = dict()

    for docid, text_body in documents:

        list_of_tokens = tokenizer(text_body)

        if use_stemming:
            list_of_tokens = stem(list_of_tokens)

        term_frequency = Counter(list_of_tokens)

        for token in term_frequency:

            if token in index:
                index[token] += [[docid, term_frequency[token]]]
            else:
                index[token] = [[docid, term_frequency[token]]]

    return index


def search(query, index, tokenizer, use_stemming):
    """
    Returns the ids of the documents containing all of the query terms,
    using the inverted index. The query goes through the same
    tokenization, normalization, and (optionally) stemming as the
    documents did, which makes the search case-insensitive. No ranking
    is performed.

    :param query: the query string
    :param index: the inverted index
    :param tokenizer: the Tokenizer the index was built with
    :param use_stemming: whether the index was built with stemming
    :return: a sorted list of matching docids
    """

    query_tokens = []

    for query_term in query.split():
        query_tokens.extend(tokenizer(query_term))

    if use_stemming:
        query_tokens = stem(query_tokens)

    doc_sets = []

    for token in query_tokens:

        postings = index.get(token, [])

        doc_sets.append(
            {docid for docid, tf in postings}
        )

    if not doc_sets:
        return []

    matches = set.intersection(*doc_sets)

    return sorted(matches, key=int)


def main():
    parser = argparse.ArgumentParser(
        description="Builds an inverted index over an XML document "
                    "collection and runs a few example queries"
    )

    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        help="Path to the XML collection"
    )

    parser.add_argument(
        "-t",
        "--tokenizer",
        choices=list(TOKENIZER_MAP.keys()),
        default="whitespace",
        help="Tokenization strategy"
    )

    parser.add_argument(
        "-s",
        "--stem",
        action="store_true",
        help="Flag for whether to stem"
    )

    args = parser.parse_args()

    documents = parse_collection(args.filepath)

    print(
        f"Parsed {len(documents)} documents from {args.filepath}"
    )

    tokenizer = TOKENIZER_MAP[args.tokenizer]()

    index = build_index(
        documents,
        tokenizer,
        args.stem
    )

    print(
        f"Built index with tokenizer={args.tokenizer}, "
        f"stemming={args.stem}: {len(index)} unique terms\n"
    )

    for query in ("system", "compatibility", "system compatibility"):

        results = search(
            query,
            index,
            tokenizer,
            args.stem
        )

        print(
            f"Query {query!r} -> docids: {results}"
        )


if __name__ == "__main__":
    main()
