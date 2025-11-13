import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Knowledge Base",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%; border-radius: 8px; height: 3em; font-weight: 500;}
    .result-card {background-color: #ffffff; padding: 1.5rem; border-radius: 8px; 
                  border-left: 4px solid #4CAF50; margin-bottom: 1rem; 
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                  color: #212121; font-size: 1.05rem; line-height: 1.6;}
    .source-badge {background-color: #2196F3; color: white; padding: 0.4rem 1rem; 
                   border-radius: 6px; font-size: 0.9rem; font-weight: 600; 
                   display: inline-block; margin-right: 0.5rem;}
    .score-badge {background-color: #FF9800; color: white; padding: 0.4rem 1rem; 
                  border-radius: 6px; font-size: 0.9rem; font-weight: 600; 
                  display: inline-block;}
    .qa-source-badge {background-color: #673AB7; color: white; padding: 0.5rem 1.2rem; 
                      border-radius: 6px; font-size: 1rem; font-weight: 600; 
                      display: inline-block; margin: 0.3rem 0;}
    .qa-score-badge {background-color: #009688; color: white; padding: 0.5rem 1.2rem; 
                     border-radius: 6px; font-size: 1rem; font-weight: 600; 
                     display: inline-block; margin: 0.3rem 0;}
    </style>
""", unsafe_allow_html=True)

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def ingest_text(content: str, file_path: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest/text",
            json={"file_path": file_path, "content": content},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def semantic_search(query: str, top_k: int = 5):
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def ask_question(question: str, top_k: int = 5):
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/qa",
            json={"question": question, "top_k": top_k},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def check_completeness(topic: str, num_docs: int = 10):
    try:
        response = requests.post(
            f"{API_BASE_URL}/query/completeness",
            json={"topic": topic, "num_docs": num_docs},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("AI Knowledge Base")
    
    if not check_api_health():
        st.error("Backend API is not running!")
        st.info("Please start the FastAPI server: `python main.py`")
        st.stop()
    
    top_k = 5
    
    tab1, tab2, tab3, tab4 = st.tabs(["Ingest", "Search", "Questions", "Analysis"])
    
    with tab1:
        st.header("Document Ingestion")
        
        document_name = st.text_input("Document Name", "my_document.txt")
        document_content = st.text_area("Document Content", height=300, 
                                       placeholder="Paste your document content here...")
        
        if st.button("Ingest Document", type="primary"):
            if document_content and document_name:
                with st.spinner("Processing document..."):
                    result = ingest_text(document_content, document_name)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        status = result.get("status", "unknown")
                        message = result.get('message', 'Operation completed')
                        chunks_count = result.get('chunks_count', 0)
                        
                        if status == "created":
                            st.success(f"✅ New Document Created")
                            st.info(f"📄 {message}")
                            st.info(f"📊 Indexed {chunks_count} chunks")
                        elif status == "indexed":
                            st.success(f"🔄 Document Updated")
                            st.info(f"📄 {message}")
                            st.info(f"📊 Indexed {chunks_count} chunks")
                        elif status == "skipped":
                            st.warning(f"⏭️ Document Unchanged")
                            st.info(f"📄 {message}")
                            st.info(f"🕐 Last indexed: {result.get('last_indexed_at', 'Unknown')}")
                        else:
                            st.info(f"Status: {status}")
                            st.info(f"📄 {message}")
                            if chunks_count > 0:
                                st.info(f"📊 Indexed {chunks_count} chunks")
            else:
                st.warning("Please provide both document name and content")
        
        uploaded_file = st.file_uploader("Or upload a file", type=['txt', 'md', 'csv'])
        
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode('utf-8')
            st.text_area("Preview", file_content[:500] + "...", height=150, disabled=True)
            
            if st.button("Ingest Uploaded File", type="primary", key="upload_btn"):
                with st.spinner("Processing file..."):
                    result = ingest_text(file_content, uploaded_file.name)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        status = result.get("status", "unknown")
                        message = result.get('message', 'Operation completed')
                        chunks_count = result.get('chunks_count', 0)
                        
                        if status == "created":
                            st.success(f"✅ New File Ingested")
                            st.info(f"📄 {message}")
                            st.info(f"📊 Indexed {chunks_count} chunks")
                        elif status == "indexed":
                            st.success(f"🔄 File Updated")
                            st.info(f"📄 {message}")
                            st.info(f"📊 Indexed {chunks_count} chunks")
                        elif status == "skipped":
                            st.warning(f"⏭️ File Unchanged")
                            st.info(f"📄 {message}")
                            st.info(f"🕐 Last indexed: {result.get('last_indexed_at', 'Unknown')}")
                        else:
                            st.info(f"Status: {status}")
                            st.info(f"📄 {message}")
                            if chunks_count > 0:
                                st.info(f"📊 Indexed {chunks_count} chunks")
    
    with tab2:
        st.header("Semantic Search")
        
        search_query = st.text_input("Enter your search query", 
                                     placeholder="e.g., How do neural networks learn?")
        
        if st.button("Search", type="primary") and search_query:
            with st.spinner("Searching..."):
                results = semantic_search(search_query, top_k)
                
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                elif "results" in results and results["results"]:
                    st.success(f"Found {len(results['results'])} relevant results")
                    
                    for idx, result in enumerate(results["results"], 1):
                        st.markdown(f"### Result {idx}")
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f'<span class="source-badge">{result.get("source", "Unknown")}</span>', 
                                      unsafe_allow_html=True)
                        with col_b:
                            score = result.get("score", 0)
                            st.markdown(f'<span class="score-badge">Score: {score:.3f}</span>', 
                                      unsafe_allow_html=True)
                        
                        st.text_area(f"Result {idx} content", result.get("text", ""), 
                                   height=150, disabled=True, label_visibility="collapsed")
                        st.markdown("---")
                else:
                    st.warning("No results found.")
    
    with tab3:
        st.header("Ask Questions")
        
        question = st.text_area("Your Question", height=100,
                               placeholder="e.g., What are the main types of machine learning?")
        
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("Thinking..."):
                result = ask_question(question, top_k)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                elif "answer" in result:
                    st.markdown("### Answer")
                    st.markdown(f'<div class="result-card">{result["answer"]}</div>', 
                              unsafe_allow_html=True)
                    
                    if "sources" in result and result["sources"]:
                        st.markdown("### Sources Used")
                        for idx, source in enumerate(result["sources"], 1):
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f'<span class="qa-source-badge">{idx}. {source.get("source", "Unknown")}</span>', 
                                          unsafe_allow_html=True)
                            with col_b:
                                score = source.get("score", 0)
                                st.markdown(f'<span class="qa-score-badge">Score: {score:.3f}</span>', 
                                          unsafe_allow_html=True)
    
    with tab4:
        st.header("Knowledge Base Analysis")
        
        topic = st.text_input("Topic to Analyze", 
                             placeholder="e.g., machine learning, computer vision")
        
        num_docs = st.slider("Number of documents to retrieve", 5, 20, 10)
        
        if st.button("Analyze Coverage", type="primary") and topic:
            with st.spinner("Analyzing..."):
                result = check_completeness(topic, num_docs)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                elif "coverage" in result:
                    coverage = result["coverage"]
                    if coverage == "comprehensive":
                        st.success(f"Coverage: Comprehensive")
                    elif coverage == "partial":
                        st.warning(f"Coverage: Partial")
                    else:
                        st.error(f"Coverage: Limited")
                    
                    st.metric("Relevant Documents Found", result.get("num_documents", 0))
                    st.markdown("### Analysis")
                    st.markdown(f'<div class="result-card">{result.get("analysis", "No analysis available")}</div>', 
                              unsafe_allow_html=True)

if __name__ == "__main__":
    main()
