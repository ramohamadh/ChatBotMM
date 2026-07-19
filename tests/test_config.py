"""Tests that config paths resolve to the project-root data/ directory."""


from chatbot import config


def test_data_paths_point_under_project_root():
    # docs/ and vectorstore/ must live under <root>/data, not inside the package.
    assert config.DATA_DIR.name == "data"
    assert config.DOCS_DIR == config.DATA_DIR / "docs"
    assert config.VECTORSTORE_DIR == config.DATA_DIR / "vectorstore"
    # The package dir must NOT be an ancestor of the data dir
    assert config.PACKAGE_DIR not in config.DATA_DIR.parents


def test_generative_defaults():
    assert config.USE_GENERATIVE is True
    assert "Qwen" in config.GENERATIVE_MODEL
