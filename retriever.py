"""
Retriever module for semantic search and RAG-based question answering.

This module implements the core retrieval and generation pipeline using
Qdrant for semantic search and LLM for answer generation.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SearchResult:
    """
    Represents a single search result from the knowledge base.
    """
    text: str
    source: str
    score: float
    chunk_index: int
    metadata: Dict[str, Any]


class LLMClient:
    """
    Wrapper for LLM API calls with fallback to mock responses.
    Supports OpenAI, Google Gemini, and other providers.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key for the LLM provider.
            model: Model name to use.
        """
        self.provider = os.getenv("LLM_PROVIDER").lower()
        self.model = model or os.getenv("LLM_MODEL")
        self.use_mock = False
        self.client = None
        
        # Initialize based on provider
        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key or self.api_key == "your-gemini-api-key-here":
                self.use_mock = True
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(self.model)
                    print(f"✓ Gemini client initialized with model: {self.model}")
                except ImportError:
                    print("Warning: google-generativeai package not installed. Using mock responses.")
                    self.use_mock = True
                except Exception as e:
                    print(f"Warning: Failed to initialize Gemini client: {e}")
                    self.use_mock = True
        else:  # openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key or self.api_key == "your-openai-api-key-here":
                self.use_mock = True
            else:
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=self.api_key)
                except ImportError:
                    print("Warning: openai package not installed. Using mock responses.")
                    self.use_mock = True
                
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt.
            max_tokens: Maximum tokens in the response.
            
        Returns:
            Generated text response.
        """
        if self.use_mock:
            return self._generate_mock_response(prompt)
        
        try:
            if self.provider == "gemini":
                # Gemini API call
                generation_config = {
                    "max_output_tokens": max_tokens,
                    "temperature": 0.7,
                }
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
            else:
                # OpenAI API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert Q&A system."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return self._generate_mock_response(prompt)
            
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response when API is not available.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            Mock response string.
        """
        if "gap" in prompt.lower() or "missing" in prompt.lower() or "completeness" in prompt.lower():
            return (
                "Based on the provided context, the following gaps or missing information have been identified:\n\n"
                "1. **Detailed implementation examples**: While concepts are mentioned, specific code examples are limited.\n"
                "2. **Performance benchmarks**: No performance metrics or benchmarking data is provided.\n"
                "3. **Troubleshooting guide**: Common issues and their solutions are not documented.\n"
                "4. **Advanced use cases**: Complex scenarios and edge cases need more coverage.\n\n"
                "These areas could benefit from additional documentation to provide a more comprehensive knowledge base."
            )
        else:
            return (
                "Based on the provided context from the knowledge base, I can provide the following information:\n\n"
                "[This is a mock response as no LLM API key is configured. The system would normally generate "
                "a detailed answer based on the retrieved context chunks.]\n\n"
                "The retrieved context provides relevant information that addresses your question. "
                "For production use, please configure a valid LLM API key in the .env file."
            )


class RAGRetriever:
    """
    Handles semantic search and RAG-based question answering.
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        collection_name: str,
        llm_client: Optional[LLMClient] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAGRetriever.
        
        Args:
            qdrant_client: Qdrant client instance.
            collection_name: Name of the Qdrant collection.
            llm_client: LLM client instance (creates default if None).
            embedding_model_name: Name of the sentence-transformers model.
        """
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.llm_client = llm_client or LLMClient()
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: The query text.
            
        Returns:
            Embedding vector.
        """
        embedding = self.embedding_model.encode(
            query,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding.tolist()
        
    def _format_search_results(self, scored_points: List[ScoredPoint]) -> List[SearchResult]:
        """
        Format Qdrant search results into SearchResult objects.
        
        Args:
            scored_points: List of scored points from Qdrant.
            
        Returns:
            List of SearchResult objects.
        """
        results = []
        for point in scored_points:
            result = SearchResult(
                text=point.payload.get("text", ""),
                source=point.payload.get("source", "unknown"),
                score=point.score,
                chunk_index=point.payload.get("chunk_index", 0),
                metadata=point.payload
            )
            results.append(result)
        return results
        
    def semantic_search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform semantic search to retrieve relevant chunks.
        
        Args:
            query: The search query.
            top_k: Number of top results to return.
            
        Returns:
            List of SearchResult objects ranked by relevance.
        """
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format and return results
        return self._format_search_results(search_results)
        
    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a question using RAG (Retrieval-Augmented Generation).
        
        This method:
        1. Performs semantic search to retrieve relevant context
        2. Constructs a prompt with the context and question
        3. Generates an answer using the LLM
        
        Args:
            question: The user's question.
            top_k: Number of context chunks to retrieve.
            
        Returns:
            Dictionary containing the answer and supporting evidence.
        """
        # Retrieve relevant context
        search_results = self.semantic_search(question, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "The knowledge base does not contain relevant information to answer this question.",
                "context_used": [],
                "sources": [],
                "num_sources": 0
            }
        
        # Construct context string
        context_parts = []
        sources = []
        for idx, result in enumerate(search_results, 1):
            context_parts.append(f"[{idx}] {result.text}")
            sources.append({
                "source": result.source,
                "chunk_index": result.chunk_index,
                "score": result.score
            })
        
        context = "\n\n".join(context_parts)
        
        # Construct prompt
        prompt = f"""You are an expert Q&A system. Use ONLY the provided context to answer the user's question. 
If the information is not in the context, state 'The knowledge base does not contain the answer.'

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        # Generate answer using LLM
        answer = self.llm_client.generate_response(prompt)
        
        return {
            "answer": answer,
            "context_used": [result.text for result in search_results],
            "sources": sources,
            "num_sources": len(sources)
        }
        
    def completeness_check(self, topic: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze the completeness of knowledge base coverage for a topic.
        
        This method:
        1. Retrieves relevant context about the topic
        2. Asks the LLM to identify gaps and missing information
        
        Args:
            topic: The topic to analyze.
            top_k: Number of context chunks to retrieve.
            
        Returns:
            Dictionary containing gap analysis and recommendations.
        """
        # Retrieve relevant context
        search_results = self.semantic_search(topic, top_k=top_k)
        
        if not search_results:
            return {
                "topic": topic,
                "analysis": f"No information found about '{topic}' in the knowledge base.",
                "coverage": "none",
                "num_documents_found": 0,
                "gaps_identified": [
                    f"No documentation exists for '{topic}'",
                    "Recommend adding comprehensive documentation on this topic"
                ],
                "context_reviewed": []
            }
        
        # Construct context string
        context_parts = []
        for idx, result in enumerate(search_results, 1):
            context_parts.append(f"[{idx}] Source: {result.source}\n{result.text}")
        
        context = "\n\n".join(context_parts)
        
        # Construct prompt for gap analysis
        prompt = f"""You are a knowledge base analyst. Review the provided context about "{topic}" and identify:
1. What information IS covered
2. What information is MISSING or incomplete
3. Any gaps in coverage or areas that need more detail
4. Recommendations for improvement

Be specific and structured in your analysis.

CONTEXT ABOUT "{topic}":
{context}

ANALYSIS:"""
        
        # Generate analysis using LLM
        analysis = self.llm_client.generate_response(prompt, max_tokens=800)
        
        # Determine coverage level
        coverage = "partial"
        if len(search_results) >= 8:
            coverage = "good"
        elif len(search_results) <= 2:
            coverage = "limited"
        
        return {
            "topic": topic,
            "analysis": analysis,
            "coverage": coverage,
            "num_documents_found": len(search_results),
            "context_reviewed": [
                {
                    "source": result.source,
                    "score": result.score,
                    "snippet": result.text[:200] + "..." if len(result.text) > 200 else result.text
                }
                for result in search_results
            ]
        }
        
    def get_similar_documents(self, reference_text: str, top_k: int = 5) -> List[SearchResult]:
        """
        Find documents similar to a reference text.
        
        Args:
            reference_text: The reference text to compare against.
            top_k: Number of similar documents to return.
            
        Returns:
            List of similar SearchResult objects.
        """
        return self.semantic_search(reference_text, top_k=top_k)


def create_rag_retriever(
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    collection_name: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None
) -> RAGRetriever:
    """
    Factory function to create a RAGRetriever instance.
    
    Args:
        qdrant_host: Qdrant host (defaults to env variable).
        qdrant_port: Qdrant port (defaults to env variable).
        collection_name: Collection name (defaults to env variable).
        llm_api_key: LLM API key (defaults to env variable).
        llm_model: LLM model name (defaults to env variable).
        
    Returns:
        Configured RAGRetriever instance.
    """
    qdrant_host = qdrant_host or os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(qdrant_port or os.getenv("QDRANT_PORT", 6333))
    collection_name = collection_name or os.getenv("QDRANT_COLLECTION_NAME", "knowledge_base")
    
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
    llm_client = LLMClient(api_key=llm_api_key, model=llm_model)
    
    return RAGRetriever(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        llm_client=llm_client
    )
