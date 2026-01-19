"""
Agent 3: Policy Retrieval Agent
RAG-based retrieval of relevant policy clauses grounded in actual policy text.
"""

from typing import Dict, List
import os


class PolicyRetrievalAgent:
    """
    Agent 3: Retrieves relevant policy clauses using RAG.
    
    Wraps the RAG system and provides structured retrieval based on:
    - Policy type (auto/property)
    - Scenario details (asset, event, location)
    - Section types (coverage, exclusions, conditions)
    """
    
    def __init__(self, retriever, policy_loader):
        """
        Initialize the retrieval agent.
        
        Args:
            retriever: PolicyRetriever instance (from rag module)
            policy_loader: PolicyLoader instance (from rag module)
        """
        self.retriever = retriever
        self.loader = policy_loader
    
    def retrieve_relevant_clauses(
        self,
        question: str,
        scenario_details: Dict,
        user_policies: Dict[str, str],
        policy_types_to_query: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant policy clauses based on scenario.
        
        Args:
            question: User's coverage question
            scenario_details: Structured scenario from Agent 2 (asset, event, location)
            user_policies: Dictionary of user's policies {type: filename}
            policy_types_to_query: List of policy types to search (from Agent 1)
            top_k: Number of top chunks to retrieve
        
        Returns:
            List of retrieved chunks with metadata:
                - text: Policy clause text
                - section_name: Section name
                - policy_type: auto/property
                - section_type: coverage/exclusion/condition
                - similarity: Relevance score
        """
        # Build enhanced query using structured scenario details
        enhanced_query = self._build_enhanced_query(question, scenario_details)
        
        # Retrieve from user's policies
        all_retrieved = []
        
        for policy_type in policy_types_to_query:
            if policy_type in user_policies:
                policy_file = user_policies[policy_type]
                
                # Get user-specific chunks
                user_chunks = self.loader.load_policy(policy_file)
                
                # Build temporary index for this user's policy
                from rag.retriever import PolicyRetriever
                temp_retriever = PolicyRetriever(
                    self.retriever.embedding_generator
                )
                temp_retriever.build_index(user_chunks)
                
                # Retrieve from user's policy
                results = temp_retriever.retrieve(enhanced_query, top_k=top_k)
                all_retrieved.extend(results)
        
        # Sort by similarity and filter
        all_retrieved.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top_k most relevant
        return all_retrieved[:top_k]
    
    def _build_enhanced_query(self, question: str, scenario_details: Dict) -> str:
        """
        Build an enhanced query using structured scenario details.
        
        This helps improve retrieval by adding context from the scenario interpretation.
        """
        asset = scenario_details.get("asset", "")
        event = scenario_details.get("event", "")
        location = scenario_details.get("location", "")
        
        # Build enhanced query with structured information
        enhanced_parts = [question]
        
        if asset and asset != "unknown":
            enhanced_parts.append(f"Asset: {asset}")
        
        if event and event != "unknown":
            enhanced_parts.append(f"Event: {event}")
        
        if location and location not in ["unknown", ""]:
            enhanced_parts.append(f"Location: {location}")
        
        enhanced_query = " | ".join(enhanced_parts)
        
        return enhanced_query
    
    def filter_by_section_type(
        self,
        chunks: List[Dict],
        section_types: List[str]
    ) -> List[Dict]:
        """
        Filter retrieved chunks by section type.
        
        Args:
            chunks: Retrieved chunks
            section_types: Types to keep (e.g., ["coverage", "exclusion"])
        
        Returns:
            Filtered chunks
        """
        if not section_types:
            return chunks
        
        filtered = [
            chunk for chunk in chunks
            if chunk.get("section_type", "").lower() in [st.lower() for st in section_types]
        ]
        
        return filtered if filtered else chunks  # Return all if no matches


if __name__ == "__main__":
    print("Agent 3: Policy Retrieval Agent")
    print("This agent wraps the RAG system for structured policy retrieval.")
    print("Run the full application to test retrieval in context.")
