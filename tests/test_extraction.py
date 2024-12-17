# tests/test_extraction.py
import pytest
from earnings_ai_demo.extraction import DocumentExtractor
from pathlib import Path

@pytest.fixture
def extractor():
   return DocumentExtractor()

@pytest.fixture
def test_files():
   base_path = Path(__file__).parent.parent / 'data' / 'documents'
   return {
       'pdf': base_path / 'a.pdf',
       'doc': base_path / 'b.docx',
       'txt': base_path / 'c.txt'
   }

def test_txt_extraction(extractor, test_files):
   result = extractor.extract_text(str(test_files['txt']))
   assert 'text' in result
   assert 'metadata' in result
   assert len(result['text']) > 0

def test_pdf_extraction(extractor, test_files):
   result = extractor.extract_text(str(test_files['pdf']))
   assert 'text' in result
   assert 'metadata' in result
   assert result['metadata']['file_type'] == 'pdf'

def test_unsupported_format(extractor, tmp_path):
   invalid_file = tmp_path / "test.xyz"
   invalid_file.touch()
   with pytest.raises(ValueError):
       extractor.extract_text(str(invalid_file))

def test_process_directory(extractor, test_files):
   data_dir = test_files['pdf'].parent
   results = extractor.process_directory(str(data_dir))
   assert len(results) > 0
   assert all('text' in v for v in results.values() if 'error' not in v)