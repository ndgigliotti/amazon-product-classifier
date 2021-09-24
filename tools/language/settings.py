import re


DEFAULT_TOKENIZER = re.compile(r"\b\w\w+\b").findall
"""Default tokenizer to use."""

DEFAULT_SEP = "_"
"""Default separator to use for tagging or joining words."""

CACHE_SIZE = int(1_000_000)
"""Maximum number of recent calls to keep for functions with LRU caching."""
