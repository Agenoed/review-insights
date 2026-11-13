from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from stop_words import get_stop_words

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
STOP_LANGS = ["ukrainian", "russian", "english", "polish", "spanish"]
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{3,}\b"
DOMAIN_STOP = [
    "додаток", "додатку", "застосунок", "застосунку", "програма", "програму",
    "приложение", "приложения", "приложении", "программа", "программу",
    "app", "application", "aplikacja", "aplikacji", "aplicación", "aplicacion",
    "будь", "ласка", "пожалуйста", "please",
]


def stop_words():
    words = set(DOMAIN_STOP)
    for lang in STOP_LANGS:
        try:
            words.update(get_stop_words(lang))
        except Exception:
            pass
    return sorted(words)


def make_vectorizer(min_df=5):
    return CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words=stop_words(), min_df=min_df, max_df=0.5)


@lru_cache(maxsize=1)
def get_encoder(model_name=EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed(texts, batch_size=64, show_progress=True):
    return get_encoder().encode(list(texts), batch_size=batch_size, show_progress_bar=show_progress, normalize_embeddings=True)


def reduce(embeddings, n_components=5, seed=42):
    from umap import UMAP

    umap = UMAP(n_neighbors=15, n_components=n_components, min_dist=0.0, metric="cosine", random_state=seed)
    return umap.fit_transform(embeddings)


def cluster_hdbscan(reduced, min_cluster_size=50, min_samples=10):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(reduced)


def cluster_kmeans(reduced, k, seed=42):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(reduced)


def ctfidf_words(texts, labels, top_n=10):
    df = pd.DataFrame({"text": list(texts), "label": labels})
    docs = df[df["label"] != -1].groupby("label")["text"].apply(" ".join)
    vectorizer = make_vectorizer().fit(texts)
    counts = vectorizer.transform(docs).toarray().astype(float)
    tf = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    idf = np.log(1 + counts.sum() / len(docs) / counts.sum(axis=0).clip(min=1))
    scores = tf * idf
    vocab = vectorizer.get_feature_names_out()
    return {int(label): [vocab[i] for i in scores[row].argsort()[::-1][:top_n]] for row, label in enumerate(docs.index)}


def lda_topics(texts, k, top_n=10, seed=42):
    vectorizer = make_vectorizer()
    counts = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=seed)
    labels = lda.fit_transform(counts).argmax(axis=1)
    vocab = vectorizer.get_feature_names_out()
    words = {t: [vocab[i] for i in lda.components_[t].argsort()[::-1][:top_n]] for t in range(k)}
    return labels, words


def npmi_coherence(texts, topic_words, top_n=10):
    vectorizer = make_vectorizer()
    present = (vectorizer.fit_transform(texts) > 0).astype(np.int8).tocsc()
    vocab = {word: i for i, word in enumerate(vectorizer.get_feature_names_out())}
    n_docs = present.shape[0]
    doc_freq = np.asarray(present.sum(axis=0)).ravel()
    topic_scores = []
    for words in topic_words.values():
        ids = [vocab[w] for w in words[:top_n] if w in vocab]
        pair_scores = []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                both = present[:, ids[a]].multiply(present[:, ids[b]]).sum()
                if both == 0:
                    pair_scores.append(-1.0)
                    continue
                p_both = both / n_docs
                if p_both >= 1:
                    pair_scores.append(1.0)
                    continue
                pmi = np.log(p_both / (doc_freq[ids[a]] / n_docs * doc_freq[ids[b]] / n_docs))
                pair_scores.append(pmi / -np.log(p_both))
        if pair_scores:
            topic_scores.append(np.mean(pair_scores))
    return float(np.mean(topic_scores)) if topic_scores else float("nan")


def word_coherence(topic_words, top_n=10):
    scores = []
    for words in topic_words.values():
        words = words[:top_n]
        if len(words) < 2:
            continue
        vectors = embed(words, show_progress=False)
        sims = vectors @ vectors.T
        scores.append((sims.sum() - len(words)) / (len(words) * (len(words) - 1)))
    return float(np.mean(scores)) if scores else float("nan")


def diversity(topic_words, top_n=10):
    words = [w for top in topic_words.values() for w in top[:top_n]]
    return len(set(words)) / len(words) if words else float("nan")


def silhouette(embeddings, labels, sample_size=5000, seed=42):
    labels = np.asarray(labels)
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return float("nan")
    size = min(sample_size, int(mask.sum()))
    return float(silhouette_score(embeddings[mask], labels[mask], metric="cosine", sample_size=size, random_state=seed))