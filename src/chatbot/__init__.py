"""
ChatBotMM — a fully local Persian/English RAG (Retrieval-Augmented Generation)
document question-answering system.
"""

__all__ = ["RAGPipeline"]
__version__ = "0.2.0"


def __getattr__(name: str):
    """Lazily import the pipeline (PEP 562).

    `from chatbot import RAGPipeline` still works, but merely importing the
    package (e.g. `chatbot.bootstrap` from main.py) no longer pulls in the
    heavy ML dependencies — otherwise the dependency auto-installer would
    crash on the missing packages it exists to install.
    """
    if name == "RAGPipeline":
        from .rag import RAGPipeline

        return RAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
