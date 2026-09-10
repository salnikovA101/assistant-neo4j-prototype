"""Retrieval pipeline: S1 embed → S2 ANN → S2b rerank → S3 graphs → S4 carousel → S5 dedup."""

from server.algorithm.pipeline import run

__all__ = ["run"]
