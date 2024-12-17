import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
from typing import Dict, Optional
import logging
import json

class DocumentExtractor:
    supported_extensions = {'.pdf', '.docx', '.txt'}

    def extract_text(self, file_path: str) -> Dict:
        """Extract text from supported document formats and save as JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        metadata = {
            "filename": path.name,
            "file_type": path.suffix.lower()[1:],
            "file_size": path.stat().st_size
        }

        try:
            if path.suffix.lower() == '.pdf':
                text = self._extract_pdf(path)
            elif path.suffix.lower() == '.docx':
                text = self._extract_docx(path)
            else:  # .txt
                text = path.read_text(encoding='utf-8')

            result = {
                "text": text,
                "metadata": metadata
            }

            # Save extraction to JSON file
            output_path = path.with_suffix('.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)

            return result

        except Exception as e:
            logging.error(f"Extraction failed for {file_path}: {str(e)}")
            raise

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF files."""
        text = []
        with fitz.open(path) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)

    def _extract_docx(self, path: Path) -> str:
        """Extract text from DOCX files."""
        doc = Document(path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def process_directory(self, directory_path: str) -> Dict[str, Dict]:
        """Process all supported documents in a directory."""
        results = {}
        directory = Path(directory_path)

        if not directory.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        for file_path in directory.glob('*'):
            if file_path.suffix.lower() in self.supported_extensions:
                try:
                    results[file_path.name] = self.extract_text(str(file_path))
                except Exception as e:
                    logging.error(f"Failed to process {file_path}: {str(e)}")
                    results[file_path.name] = {"error": str(e)}

        return results
