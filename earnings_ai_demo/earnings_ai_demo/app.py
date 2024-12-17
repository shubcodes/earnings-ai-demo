# earnings_ai_demo/earnings_ai_demo/app.py
import streamlit as st
import asyncio
from pathlib import Path
import yaml
import tempfile
import os
from database import DatabaseOperations
from transcription import AudioTranscriber
from embedding import EmbeddingGenerator
from extraction import DocumentExtractor
from query import QueryInterface

def load_config():
    with open('config/config.yaml') as f:
        return yaml.safe_load(f)

class EarningsAIApp:
    def __init__(self):
        config = load_config()
        self.db = DatabaseOperations(config['mongodb']['uri'])
        self.transcriber = AudioTranscriber(config['fireworks']['api_key'])
        self.embedding_gen = EmbeddingGenerator(config['fireworks']['api_key'])
        self.doc_extractor = DocumentExtractor()
        self.query_interface = QueryInterface(
            api_key=config['fireworks']['api_key'],
            database_operations=self.db
        )

    async def process_files(self, files):
        results = []
        for file in files:
            file_type = "audio" if file.type and file.type.startswith("audio") else "document"
            file_extension = Path(file.name).suffix
            
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=True) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file.flush()

                try:
                    if file_type == "audio":
                        result = await self.transcriber.transcribe_file(
                            tmp_file.name,
                            metadata={'company_ticker': 'MDB', 'filename': file.name}
                        )
                        text = result['transcription']
                        metadata = result['metadata']
                    else:
                        result = self.doc_extractor.extract_text(tmp_file.name)
                        text = result['text']
                        metadata = {**result['metadata'], 'filename': file.name}

                    embedding = self.embedding_gen.generate_document_embedding(text)
                    self.db.store_document(text=text, embeddings=embedding, metadata=metadata)
                    results.append({"filename": file.name, "status": "success"})
                except Exception as e:
                    results.append({"filename": file.name, "status": "error", "message": str(e)})

        return results

    async def query_documents(self, query_text):
        result = self.query_interface.query(query_text)
        return result['response'], result['sources']

def main():
    st.set_page_config(
        page_title="EarningsAI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
        <style>
        /* Global Styles */
        .stApp {
            background-color: #2c2c2e;
            color: #f5f5f7;
        }
        
        /* Typography */
        h1, h2, h3 {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 600;
            color: #ffffff;
        }
        
        /* Input and Text Styles */
        input, textarea {
            background-color: #3a3a3c;
            color: #f5f5f7;
            border: none;
            border-radius: 5px;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 0.25rem;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        
        /* Card Styling */
        .card {
            background: #3a3a3c;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0px 1px 3px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Chat Messages */
        .chat-message {
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .user-message {
            background: #1a1a1c;
            margin-left: 20%;
            margin-right: 10px;
            border-radius: 15px 15px 0 15px;
        }
        
        .bot-message {
            background: #2d2d30;
            margin-right: 20%;
            margin-left: 10px;
            border-radius: 15px 15px 15px 0;
        }
        
        /* Remove unwanted spacing */
        [data-testid="stMarkdownContainer"] {
            margin: 0 !important;
            padding: 0 !important;
        }
        
        .st-emotion-cache-1cvow4s {
            margin: 0 !important;
            padding: 0 !important;
        }

        /* Error and Success Messages */
        .st-alert-error {
            background-color: #000000;
            color: #ffffff;
            border-radius: 8px;
            padding: 1rem;
        }

        .st-alert-success {
            background-color: #000000;
            color: #ffffff;
            border-radius: 8px;
            padding: 1rem;
        }

        /* Form styling */
        .stForm {
            background-color: transparent !important;
            border: none !important;
        }
        
        .stForm [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }

        /* Chat container */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        /* Sources expander */
        .streamlit-expanderHeader {
            background-color: #3a3a3c !important;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = set()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("EarningsAI")
    st.markdown("Transform your documents into insights")

    app = EarningsAIApp()

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col2:
        # File Upload Section
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drop your files here",
            type=['pdf', 'docx', 'txt', 'mp3', 'wav'],
            accept_multiple_files=True,
            key="file_uploader"
        )

        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            if new_files and st.button("Process Files", type="primary"):
                with st.spinner("Processing files..."):
                    results = asyncio.run(app.process_files(new_files))
                    
                    for result in results:
                        if result["status"] == "success":
                            st.session_state.processed_files.add(result["filename"])
                            st.success(f"✅ {result['filename']} processed successfully")
                        else:
                            st.error(f"❌ Error processing {result['filename']}: {result['message']}")

        # Display processed files
        if st.session_state.processed_files:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📚 Processed Documents")
            for filename in st.session_state.processed_files:
                st.text(f"• {filename}")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    with col1:
        # Chat Section
        st.subheader("💬 Chat with Your Documents")
        
        # Display chat history
        for chat in st.session_state.chat_history:
            # Question
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {chat['question']}
                </div>
            """, unsafe_allow_html=True)
            
            # Answer
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Assistant:</strong><br>{chat['answer']}
                </div>
            """, unsafe_allow_html=True)
            
            # Sources
            if chat['sources']:
                with st.expander("View Sources"):
                    for source in chat['sources']:
                        st.markdown(f"""
                        **{source['metadata'].get('filename')}**  
                        Confidence: {source.get('score', 'N/A'):.2f}
                        """)
                        st.markdown("---")

        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            query = st.text_input(
                "",
                placeholder="Ask anything about your documents...",
                key="query_input"
            )
            submit_button = st.form_submit_button("Send", type="primary")

        if submit_button and query:
            with st.spinner("Analyzing..."):
                response, sources = asyncio.run(app.query_documents(query))
                
                # Add the new Q&A pair to chat history
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": response,
                    "sources": sources
                })
                
                st.rerun()

if __name__ == "__main__":
    main()