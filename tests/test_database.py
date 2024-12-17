import pytest
from unittest.mock import MagicMock
from earnings_ai_demo.database import DatabaseOperations

@pytest.fixture
def db_ops():
    return DatabaseOperations('mock_uri')

def test_store_document(db_ops):
    db_ops.documents.insert_one = MagicMock(return_value=MagicMock(inserted_id='doc_id'))
    
    doc_id = db_ops.store_document('text', [0.1, 0.2], {'key': 'value'})
    
    assert doc_id == 'doc_id'
    db_ops.documents.insert_one.assert_called_once()
        
def test_query_similar(db_ops):
    mock_results = [{'text': 'doc1'}, {'text': 'doc2'}] 
    db_ops.documents.aggregate = MagicMock(return_value=mock_results)
    
    results = db_ops.query_similar([0.1, 0.2], limit=2, filters={'key': 'value'})
    
    assert results == mock_results
    db_ops.documents.aggregate.assert_called_once()
        
def test_get_document_by_id(db_ops):
    db_ops.documents.find_one = MagicMock(return_value={'text': 'doc'})
    
    doc = db_ops.get_document_by_id('doc_id')
    
    assert doc == {'text': 'doc'}
    db_ops.documents.find_one.assert_called_once_with({'_id': 'doc_id'})
        
def test_update_document(db_ops):
    db_ops.documents.update_one = MagicMock(return_value=MagicMock(modified_count=1))
    
    updated = db_ops.update_document('doc_id', {'key': 'new_value'})
    
    assert updated == True
    db_ops.documents.update_one.assert_called_once()
        
def test_delete_document(db_ops):
    db_ops.documents.delete_one = MagicMock(return_value=MagicMock(deleted_count=1))
    
    deleted = db_ops.delete_document('doc_id')
    
    assert deleted == True
    db_ops.documents.delete_one.assert_called_once_with({'_id': 'doc_id'})