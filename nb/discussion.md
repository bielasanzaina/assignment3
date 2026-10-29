# Assignment 3 — Discussion

## Problem 1

### (A) Comparing my Naive Bayes with the posted attempt

Looking at the posted `nb-solution.py` next to my Assignment 2 code, the
overall pipeline is the same (build the DataFrames, compute priors, build a
bag-of-words matrix, smooth the likelihoods, score test documents in log
space), but there are a few differences worth pointing out.

**Things done differently in the posted attempt.** The vocabulary is built
with a set comprehension over the joined texts, so word indices can come out
in a different order between runs; I build the vocabulary in insertion order,
which makes debugging and testing easier because a word keeps the same column
every run. In `train_nb`, the posted `word_counts_per_class` sums an entire
class's submatrix down to a single scalar (the total number of tokens in the
class). That is the right denominator value, but the name suggests per-word
counts, and the per-word counts then have to be recomputed inside the
likelihood loop with `np.where` and `np.squeeze` for every (word, class)
pair, which rescans the training matrix |V| × C times. I compute the
per-class word counts once with a row sum over the class submatrix, which is
both clearer and much faster on the speech data.

**The smoothing denominator (and the shape change the assignment mentions).**
The posted likelihoods are computed as
`(alpha + count) / (alpha * (len(vocabulary) + 1) + word_counts_per_class)`.
The `+ 1` implicitly reserves a slot for an unseen word, which only forms a
proper probability distribution if the likelihood matrix actually has
`|V| + 1` columns. With a matrix of shape `(n_classes, |V|)`, the rows do not
sum to 1 under that denominator, and the values no longer match the worked
example in Jurafsky and Martin (for instance, P(predictable|−) would come out
as 2/35 instead of the textbook's 2/34). This is the change I had to make for
this problem: my `NaiveBayes` class keeps the likelihood matrix at shape
`(n_classes, |V|)` and uses `alpha * |V|` in the denominator, matching the
textbook algorithm, which drops unknown test words rather than reserving
probability mass for them. The third unit test below verifies that each
class's row sums to exactly 1, and the second verifies the textbook values.

**Smaller points.** The posted code extracts the unlabeled author with
`str(f).split('_')[-1][:-4]`, which breaks if any directory on the path
contains an underscore; `f.stem.split("_")[-1]` only looks at the filename.
The `__main__` block also computes the scikit-learn metrics without printing
them and only plots the custom model's confusion matrix, without saving it.

**What I could have done better.** My own likelihood computation still loops
over every (class, word) pair; the whole matrix can be computed in one
vectorized numpy expression, `(counts + alpha) / denominator[:, None]`. My
whitespace tokenization is also case-sensitive while `CountVectorizer`
lowercases by default, so my custom model and the scikit-learn model do not
see exactly the same features — a shared tokenizer (now a `Tokenizer` class
in `utils`) is the cleaner design.

### (C) pytest output

The three required unit tests (priors, likelihoods, and likelihoods summing
to 1 along each class axis) plus an end-to-end `test_nb_class` check, all
run on the worked-example documents from Jurafsky and Martin (Appendix B)
stored in `tests/data/docs.txt`, with add-1 smoothing:

```
============================= test session starts ==============================
platform darwin -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/bailasanzaina/Desktop/assignment3/nb
configfile: pytest.ini
plugins: anyio-4.14.2
collected 4 items

tests/test_naive_bayes.py ....                                           [100%]

============================== 4 passed in 0.90s ===============================
```

### (D) Running the project

`python -m nb.bin.main -f data` loads the data, trains the `NaiveBayes`
class, and prints the predictions:

```
Priors: [0.35294118 0.64705882]
Likelihood matrix shape: (2, 24390)

Predictions:
speech_14_kennedy.txt -> Kennedy
speech_9_kennedy.txt -> Kennedy
speech_11_kennedy.txt -> Kennedy
speech_5_johnson.txt -> Johnson
speech_6_johnson.txt -> Kennedy
speech_12_kennedy.txt -> Kennedy
speech_8_johnson.txt -> Kennedy
speech_10_johnson.txt -> Johnson
speech_7_johnson.txt -> Johnson
speech_13_kennedy.txt -> Kennedy
```

These are the same predictions my Assignment 2 script produced (8 of the 10
unlabeled speeches correct, with `speech_6_johnson` and `speech_8_johnson`
misclassified as Kennedy), which is what refactoring should look like: the
software is reorganized without altering any external behavior.

## Problem 2

The indexer is run as, e.g.:

```
python index.py -f 20newsgroups-initial.xml -t whitespace -s
```

with `-t` one of `whitespace`, `off-the-shelf` (the nltk Treebank word
tokenizer), or `ngram` (character 4-grams), and `-s` an optional flag for
Porter stemming. The program also parses `20newsgroups-additional.xml`
without any changes, so the code is not tied to the particular documents in
the initial file.

All answers below are on `20newsgroups-initial.xml` (20 documents).

### (B) Tokenizer interface

The universal property of a tokenizer is that it takes raw text and returns a
sequence of tokens. I represented that behavior with a base `Tokenizer` class
that defines `tokenize()`. The base class also handles the normalization that is
shared by all three tokenizers: lowercasing tokens and stripping punctuation.
The concrete tokenizer classes differ only in how they split the text: on
whitespace, with NLTK's Treebank tokenizer, or into character n-grams.

### (E) Query "system"

| Index | docids returned |
|---|---|
| whitespace | 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638 |
| off-the-shelf | 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638 |
| ngram (n=4) | 3, 1127, 2552, 3319, 5653, 8772, 8935, 14398, 16515, 16638 |

All three indexes return the same ten documents. The query is
case-insensitive because every token (and the query itself) is normalized by
lowercasing and stripping edge punctuation before lookup, so occurrences
like "System" and "system." all collapse to the same index entry. For the
n-gram index, the query is broken into its character 4-grams ("syst",
"yste", "stem") and a document matches if it contains all of them; here that
yields exactly the same set, but in general the n-gram index can also match
documents that only contain words like "systems" or even "ecosystem", since
the grams do not know about word boundaries.

### (F) Query "compatibility"

| Index | docids returned |
|---|---|
| whitespace | 8772, 14398 |
| off-the-shelf | 8772, 14398 |
| ngram (n=4) | 8772, 14398 |

Two documents are returned by every index. Normalization matters here too:
document 8772 contains "Compatibility" with a capital C, so without
lowercasing it would be missed. With stemming turned on, the query stems to
"compat", which would additionally match documents containing forms like
"compatible" — in this collection no other document contains any such form,
so the result set happens to stay the same.

### (G) Query "system" and "compatibility"

| Index | docids returned |
|---|---|
| whitespace | 8772, 14398 |
| off-the-shelf | 8772, 14398 |
| ngram (n=4) | 8772, 14398 |

The search treats "system" and "compatibility" as two query terms and
intersects their matching document sets, so it is a boolean AND search with
no ranking. Both documents 8772 and 14398 contain both terms, so all three
indexes return the same two document ids. For the character n-gram index,
each query term is converted to 4-grams separately before the posting lists
are intersected. This avoids creating n-grams across the space between the
two query terms, which would incorrectly turn the AND query into something
closer to a phrase search.
