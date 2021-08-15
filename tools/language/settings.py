import nltk

DEFAULT_TOKENIZER = nltk.word_tokenize
"""Default tokenizer to use when specifying tokenizer is optional."""

DEFAULT_SEP = "_"
"""Default separator to use for tagging words."""

CACHE_SIZE = int(1_000_000)
"""Maximum number of recent calls to keep for functions with LRU caching."""
