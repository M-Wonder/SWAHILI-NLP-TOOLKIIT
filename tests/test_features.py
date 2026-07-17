from swahili_nlp.features import build_Tfidf_vectorizer


def test_vectorizer_fits_and_transforms() -> None:
    vectorizer = build_Tfidf_vectorizer(max_features=100)
    corpus = [
        "Rais ametangaza sera mpya ya elimu",
        "Timu ya taifa imeshinda mechi ya kirafiki",
        "Benki kuu imeongeza thamani ya sarafu",
    ]
    matrix = vectorizer.fit_transform(corpus)
    assert matrix.shape[0] == 3
    assert matrix.shape[1] > 0


def test_vectorizer_respects_max_features() -> None:
    vectorizer = build_Tfidf_vectorizer(max_features=5)
    corpus = [
        "Rais ametangaza sera mpya ya elimu leo",
        "Timu ya taifa imeshinda mechi ya kirafiki jana",
        "Benki kuu imeongeza thamani ya sarafu wiki hii",
    ]
    matrix = vectorizer.fit_transform(corpus)
    assert matrix.shape[1] <= 5


def test_vectorizer_applies_custom_tokenizer_stopword_removal() -> None:
    vectorizer = build_Tfidf_vectorizer()
    vectorizer.fit(["Rais na waziri wa mambo ya ndani"])
    vocab = set(vectorizer.vocabulary_.keys())
    # single-token stopwords like "na"/"wa"/"ya" should not appear as unigrams
    assert "na" not in vocab
