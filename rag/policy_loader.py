"""
Policy document loader for RAG system.
Loads policy markdown files and chunks them semantically by section.
"""

import os
from typing import List, Dict
import re


class PolicyChunk:
    """Represents a chunk of policy text with metadata."""
    
    def __init__(self, text: str, policy_type: str, policy_file: str, 
                 section_name: str, chunk_type: str):
        self.text = text
        self.policy_type = policy_type  # "auto" or "property"
        self.policy_file = policy_file
        self.section_name = section_name
        self.chunk_type = chunk_type  # "coverage", "exclusion", "general"
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy serialization."""
        return {
            "text": self.text,
            "policy_type": self.policy_type,
            "policy_file": self.policy_file,
            "section_name": self.section_name,
            "chunk_type": self.chunk_type
        }


class PolicyLoader:
    """Loads and chunks policy documents."""
    
    def __init__(self, policies_dir: str):
        self.policies_dir = policies_dir
        
    def load_policy(self, policy_filename: str) -> List[PolicyChunk]:
        """
        Load a single policy file and chunk it by sections.
        
        Args:
            policy_filename: Name of the policy file (e.g., "auto_policy_1.md")
            
        Returns:
            List of PolicyChunk objects
        """
        filepath = os.path.join(self.policies_dir, policy_filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Determine policy type from filename
        policy_type = "auto" if "auto" in policy_filename.lower() else "property"
        
        chunks = []
        
        # Split by main sections (marked by ##)
        sections = re.split(r'\n## ', content)
        
        for i, section in enumerate(sections):
            if i == 0:
                # First section contains header info - keep as general info
                if section.strip():
                    chunks.append(PolicyChunk(
                        text=section.strip(),
                        policy_type=policy_type,
                        policy_file=policy_filename,
                        section_name="Policy Header",
                        chunk_type="general"
                    ))
                continue
            
            # Add back the ## marker
            section = "## " + section
            
            # Extract section title
            section_title_match = re.match(r'## (.+?)\n', section)
            section_title = section_title_match.group(1) if section_title_match else "Unknown Section"
            
            # Determine chunk type based on section content
            chunk_type = self._determine_chunk_type(section_title, section)
            
            # Split large sections into subsections (marked by ###)
            subsections = re.split(r'\n### ', section)
            
            if len(subsections) == 1:
                # No subsections, use entire section as one chunk
                chunks.append(PolicyChunk(
                    text=section.strip(),
                    policy_type=policy_type,
                    policy_file=policy_filename,
                    section_name=section_title,
                    chunk_type=chunk_type
                ))
            else:
                # Process each subsection
                for j, subsection in enumerate(subsections):
                    if j == 0:
                        # First part is section intro
                        if subsection.strip():
                            chunks.append(PolicyChunk(
                                text=subsection.strip(),
                                policy_type=policy_type,
                                policy_file=policy_filename,
                                section_name=section_title,
                                chunk_type=chunk_type
                            ))
                        continue
                    
                    # Add back the ### marker
                    subsection = "### " + subsection
                    
                    # Extract subsection title
                    subsection_title_match = re.match(r'### (.+?)\n', subsection)
                    subsection_title = subsection_title_match.group(1) if subsection_title_match else "Unknown Subsection"
                    
                    full_section_name = f"{section_title} - {subsection_title}"
                    
                    chunks.append(PolicyChunk(
                        text=subsection.strip(),
                        policy_type=policy_type,
                        policy_file=policy_filename,
                        section_name=full_section_name,
                        chunk_type=chunk_type
                    ))
        
        return chunks
    
    def _determine_chunk_type(self, section_title: str, section_content: str) -> str:
        """Determine if section is about coverage, exclusions, or general info."""
        title_lower = section_title.lower()
        content_lower = section_content.lower()
        
        # Check for exclusions
        if "exclusion" in title_lower or "not covered" in title_lower:
            return "exclusion"
        
        # Check for coverage
        if "coverage" in title_lower or "perils insured" in title_lower:
            return "coverage"
        
        # Check content for exclusion keywords
        exclusion_keywords = ["not covered", "we do not cover", "excluded", "exclusion"]
        if any(keyword in content_lower for keyword in exclusion_keywords):
            return "exclusion"
        
        # Check content for coverage keywords
        coverage_keywords = ["we will pay", "we cover", "coverage includes"]
        if any(keyword in content_lower for keyword in coverage_keywords):
            return "coverage"
        
        return "general"
    
    def load_user_policies(self, user_policies: Dict[str, str]) -> List[PolicyChunk]:
        """
        Load all policies for a specific user.
        
        Args:
            user_policies: Dictionary mapping policy types to filenames
                          e.g., {"auto": "auto_policy_1.md", "property": "property_policy_1.md"}
        
        Returns:
            List of all PolicyChunk objects for the user
        """
        all_chunks = []
        
        for policy_type, filename in user_policies.items():
            chunks = self.load_policy(filename)
            all_chunks.extend(chunks)
        
        return all_chunks


if __name__ == "__main__":
    # Test the loader
    loader = PolicyLoader("../policies")
    
    # Test loading Alice's policy
    chunks = loader.load_policy("auto_policy_1.md")
    print(f"Loaded {len(chunks)} chunks from Alice's auto policy")
    print(f"\nFirst chunk:")
    print(f"Section: {chunks[0].section_name}")
    print(f"Type: {chunks[0].chunk_type}")
    print(f"Text preview: {chunks[0].text[:200]}...")
