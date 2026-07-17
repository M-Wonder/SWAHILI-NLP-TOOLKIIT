"""
Command-line interface for the Swahili NLP toolkit.

Subcommands
-----------
train     Train a classifier on a labeled CSV file and save it to disk.
predict   Load a saved model and classify one or more headlines.
evaluate  Load a saved model, evaluate it on a labeled CSV, and print
          a classification report (optionally saving a confusion matrix).

Run ``swahili-nlp --help`` (or ``python -m swahili_nlp.cli --help``) for
full usage details.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from swahili_nlp.classifier import SwahiliNewsClassifier
from swahili_nlp.evaluate import generate_report, plot_confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("swahili_nlp.cli")

DEFAULT_MODEL_PATH = "models/swahili_news_classifier.joblib"


def _load_dataset(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset has no rows after dropping missing values")
    return df


def cmd_train(args: argparse.Namespace) -> None:
    df = _load_dataset(args.data, args.text_col, args.label_col)
    x_train, x_test, y_train, y_test = train_test_split(
        df[args.text_col].tolist(),
        df[args.label_col].tolist(),
        test_size=args.test_size,
        random_state=42,
        stratify=df[args.label_col],
    )

    clf = SwahiliNewsClassifier(max_features=args.max_features, C=args.C)
    clf.fit(x_train, y_train)

    y_pred = clf.predict_batch(x_test)
    report = generate_report(y_test, y_pred)
    print("\nHeld-out test set performance:\n")
    print(report)

    clf.save(args.model_out)
    print(f"Model saved to: {args.model_out}")

    if args.confusion_matrix:
        labels = sorted(set(y_test) | set(y_pred))
        cm_path = plot_confusion_matrix(y_test, y_pred, labels, args.confusion_matrix)
        print(f"Confusion matrix saved to: {cm_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    clf = SwahiliNewsClassifier.load(args.model)
    texts = args.text if args.text else [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        print("No input text provided (use --text or pipe text via stdin).")
        sys.exit(1)

    for text in texts:
        result = clf.predict_proba(text)
        top3 = sorted(result.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_str = ", ".join(f"{label}={prob:.2f}" for label, prob in top3)
        print(f"Text:       {text}")
        print(f"Prediction: {result.label} (confidence={result.confidence:.2f})")
        print(f"Top-3:      {top3_str}\n")


def cmd_evaluate(args: argparse.Namespace) -> None:
    clf = SwahiliNewsClassifier.load(args.model)
    df = _load_dataset(args.data, args.text_col, args.label_col)
    y_true = df[args.label_col].tolist()
    y_pred = clf.predict_batch(df[args.text_col].tolist())
    print(generate_report(y_true, y_pred))
    if args.confusion_matrix:
        labels = sorted(set(y_true) | set(y_pred))
        cm_path = plot_confusion_matrix(y_true, y_pred, labels, args.confusion_matrix)
        print(f"Confusion matrix saved to: {cm_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swahili-nlp",
        description="Train, evaluate, and run a Swahili news headline classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_p = subparsers.add_parser("train", help="Train a new model")
    train_p.add_argument("--data", default="data/swahili_news_dataset.csv",
                          help="Path to labeled CSV dataset")
    train_p.add_argument("--text-col", default="text")
    train_p.add_argument("--label-col", default="category")
    train_p.add_argument("--model-out", default=DEFAULT_MODEL_PATH,
                          help="Where to save the trained model")
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--max-features", type=int, default=5000)
    train_p.add_argument("--C", type=float, default=5.0,
                          help="Inverse regularization strength for Logistic Regression")
    train_p.add_argument("--confusion-matrix", default="models/confusion_matrix.png",
                          help="Path to save a confusion matrix image (omit to skip)")
    train_p.set_defaults(func=cmd_train)

    predict_p = subparsers.add_parser("predict", help="Classify headline(s)")
    predict_p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    predict_p.add_argument("--text", action="append",
                            help="Headline to classify (repeatable). "
                                 "If omitted, reads lines from stdin.")
    predict_p.set_defaults(func=cmd_predict)

    eval_p = subparsers.add_parser("evaluate", help="Evaluate a saved model on labeled data")
    eval_p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    eval_p.add_argument("--data", default="data/swahili_news_dataset.csv")
    eval_p.add_argument("--text-col", default="text")
    eval_p.add_argument("--label-col", default="category")
    eval_p.add_argument("--confusion-matrix", default=None)
    eval_p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
