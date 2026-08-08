"""Shared test configuration.

Force HuggingFace offline mode: every model a test needs is either stubbed or
already in the local cache, and the hub's "is there a newer revision?" check
can hang the whole suite on a slow or saturated connection.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
