"""
Embeddings module for RAG system.
Uses sentence-transformers to generate embeddings for policy chunks.
"""

from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingGenerator:
    """Generates embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is a lightweight, fast model good for semantic search.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (num_texts x embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )


if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()
    
    test_texts = [
        "Am I covered if my car is stolen?",
        "Coverage for vehicle theft is provided under Comprehensive Coverage",
        "Fire damage to your home is covered under dwelling protection"
    ]
    
    embeddings = generator.generate_embeddings(test_texts)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    print(f"First embedding (first 10 values): {embeddings[0][:10]}")
