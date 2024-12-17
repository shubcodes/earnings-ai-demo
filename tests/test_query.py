# tests/test_query.py
import pytest
from earnings_ai_demo.query import QueryInterface
from unittest.mock import Mock
import fireworks.client

@pytest.fixture
def mock_components():
    embedding_gen = Mock()
    db_ops = Mock()
    embedding_gen.generate_embedding.return_value = [0.1] * 768
    db_ops.query_similar.return_value = [
        {
            'text': 'Test document content',
            'metadata': {
                'document_type': 'test',
                'filename': 'test.pdf'
            }
        }
    ]
    return embedding_gen, db_ops


@pytest.fixture
def query_interface(config, mock_components):
    embedding_gen, db_ops = mock_components
    return QueryInterface(config['fireworks']['api_key'], embedding_gen, db_ops)

def test_basic_query(query_interface):
    result = query_interface.query("Test query")
    assert 'response' in result
    assert 'sources' in result

def test_filtered_query(query_interface):
    result = query_interface.query(
        "Test query",
        company_ticker="MDB",
        doc_type="earnings_call"
    )
    assert 'response' in result

@pytest.mark.asyncio
async def test_streaming_query(query_interface):
    chunks = []
    async def callback(chunk):
        chunks.append(chunk)
    
    await query_interface.process_streaming_query(
        "Test query",
        callback
    )
    assert len(chunks) > 0