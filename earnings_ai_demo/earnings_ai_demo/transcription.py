# earnings_ai_demo/earnings_ai_demo/transcription.py
import os
from typing import Dict, Optional
import requests
from fireworks.client.audio import AudioInference
import json
import logging

class AudioTranscriber:
    def __init__(self, api_key: str, base_url: str = "https://audio-turbo.us-virginia-1.direct.fireworks.ai"):
        self.client = AudioInference(
            model="whisper-v3-turbo",
            base_url=base_url,
            api_key=api_key
        )
        
    async def transcribe_file(self, 
                            file_path: str, 
                            metadata: Optional[Dict] = None) -> Dict:
        """Transcribe an audio file and return the transcription with metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        try:
            with open(file_path, 'rb') as audio_file:
                audio_content = audio_file.read()
                
            response = await self.client.transcribe_async(
                audio=audio_content,
                language="en",
                temperature=0
            )
            
            result = {
                "transcription": response.text,
                "metadata": {
                    "filename": os.path.basename(file_path),
                    "file_size": len(audio_content),
                    "duration": getattr(response, 'duration', None),
                    **(metadata if metadata else {})
                }
            }
            
            # Save transcription to JSON file
            output_path = file_path.rsplit('.', 1)[0] + '_transcription.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            return result
            
        except Exception as e:
            logging.error(f"Transcription failed for {file_path}: {str(e)}")
            raise
            
    async def transcribe_directory(self, 
                                 directory_path: str, 
                                 metadata: Optional[Dict] = None) -> Dict[str, Dict]:
        """Transcribe all audio files in a directory."""
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
            
        results = {}
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                file_path = os.path.join(directory_path, filename)
                try:
                    results[filename] = await self.transcribe_file(
                        file_path,
                        metadata=metadata
                    )
                except Exception as e:
                    logging.error(f"Failed to transcribe {filename}: {str(e)}")
                    results[filename] = {"error": str(e)}
                    
        return results