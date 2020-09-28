#!/usr/bin/env python3
"""
Implement very simple similarity search
"""

from typing import List, Optional
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from curiosity.db import Fact, create_sql


class Similarity:
    def __init__(self, wiki_sql_path: Optional[str] = None) -> None:
        self._wiki_sql_path = wiki_sql_path
        self._vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            strip_accents='unicode',
            decode_error='ignore'
        )

    def train(self) -> None:
        if self._wiki_sql_path is None:
            raise ValueError('Cannot fit tfidf with wiki_sql_path unset')
        _, session = create_sql(self._wiki_sql_path)
        docs = [r[0] for r in session.query(Fact.text).all()]
        self._vectorizer.fit(docs)

    def save(self, tfidf_path: str) -> None:
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self._vectorizer, f)

    def load(self, tfidf_path: str) -> None:
        with open(tfidf_path, 'rb') as f:
            self._vectorizer = pickle.load(f)

    def score(self, query: str, docs: List[str]) -> List[float]:
        query_vector = self._vectorizer.transform([query])
        doc_vectors = self._vectorizer.transform(docs)
        return cosine_similarity(query_vector, doc_vectors)[0].tolist()
