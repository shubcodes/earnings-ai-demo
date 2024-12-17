import pytest
from earnings_ai_demo.embedding import EmbeddingGenerator
import numpy as np
import yaml
from pathlib import Path

@pytest.fixture
def config():
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

@pytest.fixture
def embedding_generator(config):
    return EmbeddingGenerator(config['fireworks']['api_key'])

def test_single_embedding(embedding_generator):
    text = "This is a test document"
    embedding = embedding_generator.generate_embedding(text)
    assert isinstance(embedding, list)
    assert len(embedding) == 768
    assert all(isinstance(x, float) for x in embedding)

def test_batch_embeddings(embedding_generator):
    texts = ["First document", "Second document", "Third document"]
    embeddings = embedding_generator.generate_embeddings_batch(texts)
    assert len(embeddings) == 3
    assert all(len(emb) == 768 for emb in embeddings)

def test_document_chunking(embedding_generator):
    long_text = " ".join(["document"] * 2000)
    chunks = embedding_generator._chunk_text(long_text)
    assert len(chunks) > 1
    
def test_document_embedding(embedding_generator):
    long_text = " ".join(["document"] * 2000)
    embedding = embedding_generator.generate_document_embedding(long_text)
    assert len(embedding) == 768

def test_prefix_handling(embedding_generator):
    text = "test document"
    embedding1 = embedding_generator.generate_embedding(text)
    embedding2 = embedding_generator.generate_embedding(text, prefix="query: ")
    assert not np.array_equal(embedding1, embedding2)