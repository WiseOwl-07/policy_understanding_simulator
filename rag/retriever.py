"""
Retriever module for RAG system.
Performs semantic search over policy chunks using FAISS.
"""

from typing import List, Dict, Optional
import numpy as np
import faiss
from rag.policy_loader import PolicyChunk
from rag.embeddings import EmbeddingGenerator


class PolicyRetriever:
    """Retrieves relevant policy chunks using semantic search."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.embedding_generator = embedding_generator
        self.chunks: List[PolicyChunk] = []
        self.index: Optional[faiss.Index] = None
        
    def build_index(self, chunks):
        """
        Build FAISS index from policy chunks.
        
        Args:
            chunks: List of PolicyChunk objects or dicts to index
        """
        # Convert dicts to PolicyChunk objects if needed
        processed_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                # Convert dict to PolicyChunk
                processed_chunks.append(PolicyChunk(
                    text=chunk['text'],
                    policy_type=chunk['policy_type'],
                    policy_file=chunk['policy_file'],
                    section_name=chunk['section_name'],
                    chunk_type=chunk['chunk_type']
                ))
            else:
                # Already a PolicyChunk object
                processed_chunks.append(chunk)
        
        self.chunks = processed_chunks
        
        # Extract text from chunks
        texts = [chunk.text for chunk in processed_chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} policy chunks...")
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Build FAISS index
        print("Building FAISS index...")
        embedding_dim = embeddings.shape[1]
        
        # Use L2 distance for cosine similarity (after normalization)
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 5, 
                policy_type_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve most relevant policy chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of top results to return
            policy_type_filter: Optional filter for "auto" or "property" 
                              (None returns from all policies)
        
        Returns:
            List of dictionaries containing chunk info and similarity scores
        """
        if self.index is None or len(self.chunks) == 0:
            raise ValueError("Index not built. Call build_index first.")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search - get more results than needed if filtering
        search_k = top_k * 3 if policy_type_filter else top_k
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            chunk = self.chunks[idx]
            
            # Apply policy type filter if specified
            if policy_type_filter and chunk.policy_type != policy_type_filter:
                continue
            
            # Convert L2 distance to similarity score (0-1, higher is better)
            # For normalized vectors: L2 = 2(1 - cosine_sim)
            # So: cosine_sim = 1 - L2/2
            similarity = 1 - (distance / 2)
            
            results.append({
                "chunk": chunk,
                "similarity": float(similarity),
                "text": chunk.text,
                "section_name": chunk.section_name,
                "policy_type": chunk.policy_type,
                "policy_file": chunk.policy_file,
                "chunk_type": chunk.chunk_type
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_by_policy_types(self, query: str, policy_types: List[str], 
                                 top_k_per_type: int = 3) -> Dict[str, List[Dict]]:
        """
        Retrieve results grouped by policy type.
        
        Args:
            query: User's question
            policy_types: List of policy types to search (e.g., ["auto", "property"])
            top_k_per_type: Number of results per policy type
        
        Returns:
            Dictionary mapping policy type to list of results
        """
        results_by_type = {}
        
        for policy_type in policy_types:
            results = self.retrieve(
                query=query,
                top_k=top_k_per_type,
                policy_type_filter=policy_type
            )
            results_by_type[policy_type] = results
        
        return results_by_type


if __name__ == "__main__":
    # Test the retriever
    from rag.policy_loader import PolicyLoader
    
    # Load policies
    loader = PolicyLoader("../policies")
    chunks = loader.load_user_policies({
        "auto": "auto_policy_1.md",
        "property": "property_policy_1.md"
    })
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Build retriever
    generator = EmbeddingGenerator()
    retriever = PolicyRetriever(generator)
    retriever.build_index(chunks)
    
    # Test retrieval
    query = "Am I covered if my car is stolen?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\n\nQuery: {query}")
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.3f}")
        print(f"   Policy: {result['policy_type']} - {result['section_name']}")
        print(f"   Type: {result['chunk_type']}")
        print(f"   Text preview: {result['text'][:150]}...")
