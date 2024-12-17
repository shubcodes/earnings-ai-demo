# earnings_ai_demo/earnings_ai_demo/database.py
import time
from pymongo import MongoClient, ASCENDING
from pymongo.server_api import ServerApi
from pymongo.operations import SearchIndexModel
from typing import Dict, List, Union
import logging

class DatabaseOperations:
   def __init__(self, connection_uri: str):
       self.client = MongoClient(connection_uri, server_api=ServerApi('1'))
       self.db = self.client.earnings_db
       self.documents = self.db.documents
       self.index_ready = False
       self._setup_indexes()

   def _setup_indexes(self):
       try:
           existing_indices = list(self.documents.list_search_indexes())
           if not any(idx["name"] == "vector_index" for idx in existing_indices):
               search_index_model = SearchIndexModel(
                   definition={
                       "fields": [{
                           "type": "vector",
                           "path": "embeddings",
                           "numDimensions": 768,
                           "similarity": "dotProduct"
                       }]
                   },
                   name="vector_index",
                   type="vectorSearch"
               )
               self.documents.create_search_index(model=search_index_model)
               self._wait_for_index_build()

           self.documents.create_index(
               [("metadata.filename", ASCENDING), 
                ("metadata.document_type", ASCENDING)],
               unique=True,
               background=True
           )
           
       except Exception as e:
           if "IndexAlreadyExists" not in str(e):
               logging.error(f"Index setup failed: {e}")
               raise

   def _wait_for_index_build(self, timeout=60, retry_interval=2):
       start = time.time()
       while time.time() - start < timeout:
           indices = list(self.documents.list_search_indexes())
           if indices and all(idx.get("queryable", False) for idx in indices):
               self.index_ready = True
               logging.info("Vector search index is ready")
               return True
           time.sleep(retry_interval)
       raise TimeoutError(f"Index build timed out after {timeout}s")

   def store_document(self, text: str, embeddings: List[float], metadata: Dict[str, Union[str, int]]) -> str:
       try:
           doc = {
               "text": text,
               "embeddings": embeddings,
               "metadata": metadata
           }
           result = self.documents.update_one(
               {
                   "metadata.filename": metadata["filename"],
                   "metadata.document_type": metadata.get("document_type", "unknown")
               },
               {"$set": doc},
               upsert=True
           )
           doc_id = str(result.upserted_id or result.matched_count)
           logging.info(f"Stored document: {metadata['filename']}")
           return doc_id
           
       except Exception as e:
           logging.error(f"Store failed for {metadata.get('filename')}: {e}")
           raise

   def query_similar(self, query_embedding: List[float], limit: int = 5, filters: Dict = None) -> List[Dict]:
       try:
           if not self.index_ready:
               logging.warning("Vector index not ready yet")
               self._wait_for_index_build()

           doc_count = self.documents.count_documents({})
           if doc_count == 0:
               logging.warning("No documents in collection")
               return []

           pipeline = [
               {
                   "$vectorSearch": {
                       "index": "vector_index",
                       "queryVector": query_embedding,
                       "path": "embeddings",
                       "numCandidates": limit * 10,
                       "limit": limit
                   }
               }
           ]
           
           if filters:
               pipeline.append({"$match": filters})
               
           pipeline.append({
               "$project": {
                   "text": 1,
                   "metadata": 1,
                   "score": {"$meta": "vectorSearchScore"}
               }
           })
           
           # Wait for documents to be indexed
           results = []
           max_retries = 3
           for attempt in range(max_retries):
               results = list(self.documents.aggregate(pipeline))
               if results:
                   break
               logging.info(f"No results found, attempt {attempt+1}/{max_retries}")
               time.sleep(2)
               
           logging.info(f"Found {len(results)} similar documents")
           return results
           
       except Exception as e:
           logging.error(f"Query failed: {e}")
           return []

   def is_ready(self):
       return self.index_ready and self.documents.count_documents({}) > 0