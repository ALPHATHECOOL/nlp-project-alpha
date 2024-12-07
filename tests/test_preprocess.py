# tests/test_preprocess.py
from src.preprocess import tokenize

def test_tokenize():
    assert tokenize("Hello world") == ["Hello", "world"]
