import asyncio
import yaml
from pathlib import Path
from earnings_ai_demo.database import DatabaseOperations
from earnings_ai_demo.transcription import AudioTranscriber
from earnings_ai_demo.embedding import EmbeddingGenerator
from earnings_ai_demo.extraction import DocumentExtractor
from earnings_ai_demo.query import QueryInterface
import logging


async def main():
    # Load config
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize components
    db = DatabaseOperations(config['mongodb']['uri'])
    transcriber = AudioTranscriber(config['fireworks']['api_key'])
    embedding_gen = EmbeddingGenerator(config['fireworks']['api_key'])
    doc_extractor = DocumentExtractor()
    query_interface = QueryInterface(
        api_key=config['fireworks']['api_key'],
        database_operations=db
    )

    # Process audio files
    audio_results = await transcriber.transcribe_directory(
        'data/audio',
        metadata={'company_ticker': 'MDB'}
    )

    # Process documents
    doc_results = doc_extractor.process_directory('data/documents')

    # Debugging: Log the processed results
    logging.info("Audio Results:")
    logging.info(audio_results)
    logging.info("Document Results:")
    logging.info(doc_results)

    # Process audio transcripts and create embeddings
    for source, content in audio_results.items():
        if 'error' in content:
            continue
            
        try:
            # Create embedding from transcription
            embedding = embedding_gen.generate_document_embedding(
                content['transcription'], 
                prefix="audio_transcript: "
            )
            
            # Store in MongoDB
            db.store_document(
                text=content['transcription'],
                embeddings=embedding,
                metadata={
                    **content['metadata'],
                    'document_type': 'audio_transcript',
                    'company_ticker': 'MDB'
                }
            )
            logging.info(f"Stored audio transcript: {source}")
        except Exception as e:
            logging.error(f"Failed to process audio {source}: {e}")

    # Process documents (your existing document processing code)
    for source, content in doc_results.items():
        if 'error' in content or 'text' not in content:
            continue

        try:
            embedding = embedding_gen.generate_document_embedding(
                content['text'],
                prefix="document: "
            )
            db.store_document(
                text=content['text'],
                embeddings=embedding,
                metadata={
                    **content['metadata'],
                    'document_type': 'document',
                    'company_ticker': 'MDB'
                }
            )
            logging.info(f"Stored document: {source}")
        except Exception as e:
            logging.error(f"Failed to process document {source}: {e}")

    # Test queries
    queries = [
        "What is the total q3 earnings for fiscal 2025?",
        "what is the future of AI at Mongo? Is it going to be big?"
    ]

    for query in queries:
        logging.info(f"Processing query: {query}")
        result = query_interface.query(
            query=query,
            company_ticker='MDB',
            num_results=5
        )
        print(f"\nQuery: {query}")
        print(f"Response: {result['response']}")
        print("\nSources:")
        for source in result['sources']:
            print(f"- {source['metadata'].get('filename')} ({source['metadata'].get('document_type')})")
            print(f"  Score: {source.get('score', 'N/A')}")
            print(f"  Preview: {source['text'][:200]}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
