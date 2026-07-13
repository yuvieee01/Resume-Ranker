import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def preprocess(text: str) -> str:
    """Tokenize, lemmatize, and strip stop-words / punctuation."""
    if not text:
        return ""
    doc = nlp(text)
    tokens = [
        tok.lemma_.lower()
        for tok in doc
        if tok.is_alpha and not tok.is_stop and not tok.is_punct
    ]
    return " ".join(tokens)


def compute_similarity(job_description: str, resume_text: str) -> float:
    """
    Score a single resume against a job description.

    Both inputs are raw text — preprocessing and TF-IDF vectorization
    happen internally. Returns a cosine-similarity score in [0, 1].
    """
    processed_jd = preprocess(job_description)
    processed_resume = preprocess(resume_text)

    if not processed_jd or not processed_resume:
        return 0.0

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([processed_jd, processed_resume])

    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(float(score), 4)
