import pytest
from earnings_ai_demo.transcription import AudioTranscriber
import os
import asyncio
from pathlib import Path

BASE_PATH = Path(__file__).parent.parent / 'data' / 'audio'

@pytest.fixture
def transcriber(config):
    return AudioTranscriber(config['fireworks']['api_key'])

@pytest.fixture
def test_audio_file():
    audio_path = BASE_PATH / "sample.mp3"
    if not audio_path.exists():
        pytest.fail(f"Test audio file not found at {audio_path}")
    return audio_path

@pytest.mark.asyncio
async def test_single_file_transcription(transcriber, test_audio_file):
    result = await transcriber.transcribe_file(str(test_audio_file))
    assert 'transcription' in result
    assert 'metadata' in result
    assert result['metadata']['filename'] == 'sample.mp3'

@pytest.mark.asyncio
async def test_metadata_handling(transcriber, test_audio_file):
    metadata = {'company': 'MDB', 'date': '2024-03-21'}
    result = await transcriber.transcribe_file(
        str(test_audio_file),
        metadata=metadata
    )
    assert result['metadata']['company'] == 'MDB'
    assert result['metadata']['date'] == '2024-03-21'

def test_file_not_found(transcriber):
    with pytest.raises(FileNotFoundError):
        asyncio.run(transcriber.transcribe_file('nonexistent.mp3'))
