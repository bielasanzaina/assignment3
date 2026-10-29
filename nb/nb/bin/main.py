import argparse

from nb.nb import NaiveBayes
from nb.utils.load_data import build_dataframe, AUTHOR_TO_ID_MAP


ID_TO_AUTHOR_MAP = {
    v: k.capitalize() for k, v in AUTHOR_TO_ID_MAP.items()
}


def main():
    parser = argparse.ArgumentParser(
        description="Naive Bayes authorship identification"
    )

    # (a) Ask the user for an input directory to load the data from.
    parser.add_argument(
        "-f",
        "--indir",
        required=True,
        help="Data directory"
    )

    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.1,
        help="Lidstone smoothing parameter"
    )

    args = parser.parse_args()

    # (b) Build the DataFrames from the text within the data subfolders.
    training_df, test_df = build_dataframe(
        args.indir
    )

    # (c, d) Produce estimates for the priors and likelihoods by running
    # the train method from the NaiveBayes class.
    model = NaiveBayes(
        alpha=args.alpha
    )

    model.train(training_df)

    print(
        "Priors:",
        model.priors
    )

    print(
        "Likelihood matrix shape:",
        model.likelihoods.shape
    )

    # (e) Run the test method from the NaiveBayes class.
    class_predictions = model.test(
        test_df
    )

    # (f) Print out the predictions.
    print(
        "\nPredictions:"
    )

    for filename, pred in zip(
        test_df["filename"],
        class_predictions
    ):

        print(
            filename,
            "->",
            ID_TO_AUTHOR_MAP[pred]
        )


if __name__ == "__main__":
    main()
