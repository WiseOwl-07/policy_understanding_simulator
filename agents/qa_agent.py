"""
Q&A Agent for generating coverage responses.
Uses retrieved policy chunks and LLM to generate plain-English answers.
"""

from typing import List, Dict
from groq import Groq
import os
import json


class QAAgent:
    """Generates structured coverage responses from policy chunks."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the QA agent.
        
        Args:
            api_key: Groq API key (or will use GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def generate_response(self, question: str, retrieved_chunks: List[Dict],
                         policy_types_queried: List[str]) -> Dict:
        """
        Generate a structured coverage response.
        
        Args:
            question: User's coverage question
            retrieved_chunks: List of retrieved policy chunks with metadata
            policy_types_queried: List of policy types that were queried
        
        Returns:
            Dictionary with:
                - policy_applied: "Auto" / "Property" / "Both"
                - coverage_result: "Yes" / "No" / "It depends"
                - explanation: Plain-English explanation
                - policy_references: List of specific policy sections cited
                - disclaimer: Standard disclaimer text
        """
        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)
        
        # Determine policy applied
        policy_applied = self._determine_policy_applied(
            retrieved_chunks, policy_types_queried
        )
        
        # Create prompt
        prompt = self._create_prompt(question, context, policy_applied)
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful insurance policy assistant. Your role is to explain insurance coverage in plain English based on policy documents. Always ground your answers in the provided policy text and be clear about coverage limitations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            # Add policy_applied and disclaimer
            result["policy_applied"] = policy_applied
            result["disclaimer"] = self._get_disclaimer()
            
            # Extract policy references from chunks
            result["policy_references"] = self._extract_references(retrieved_chunks)
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {result_text}")
            
            # Return a fallback response
            return {
                "policy_applied": policy_applied,
                "coverage_result": "It depends",
                "explanation": "I'm having trouble analyzing your policy. Please contact your insurance agent for specific coverage details.",
                "policy_references": self._extract_references(retrieved_chunks),
                "disclaimer": self._get_disclaimer()
            }
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Reference {i}] {chunk['section_name']} "
                f"({chunk['policy_type'].upper()} Policy):\n{chunk['text']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _determine_policy_applied(self, chunks: List[Dict], 
                                  policy_types: List[str]) -> str:
        """Determine which policy/policies apply."""
        if not chunks:
            return "Unknown"
        
        # Check which policy types appear in retrieved chunks
        chunk_types = set(chunk['policy_type'] for chunk in chunks)
        
        if len(chunk_types) == 2:
            return "Both Auto & Property"
        elif "auto" in chunk_types:
            return "Auto"
        elif "property" in chunk_types:
            return "Property"
        else:
            return "Unknown"
    
    def _create_prompt(self, question: str, context: str, 
                      policy_applied: str) -> str:
        """Create the prompt for the LLM."""
        return f"""Based on the following insurance policy excerpts, answer the user's coverage question.

User Question: "{question}"

Policy Excerpts:
{context}

Instructions:
1. Determine if the scenario is covered: "Yes", "No", or "It depends"
2. Provide a clear, scannable explanation using inline emoji indicators
3. Use these emojis within your explanation:
   - ✔️ before aspects that ARE covered or positive points
   - ❌ before aspects that are NOT covered or exclusions  
   - 📌 before important conditions, limitations, or notes
4. Keep the explanation concise (2-4 sentences) and easy to scan
5. Cite specific coverage sections when relevant

Respond in this exact JSON format:
{{
    "coverage_result": "Yes" OR "No" OR "It depends",
    "explanation": "A clear explanation using inline emojis. Example: ✔️ Your car is covered under comprehensive coverage for theft. ❌ Your policy does not cover commercial use. 📌 A $250 deductible applies."
}}

IMPORTANT: 
- Base your answer ONLY on the provided policy excerpts
- Do not make assumptions about coverage not mentioned in the excerpts
- If the policy explicitly excludes something, say "No"
- If the policy explicitly covers something, say "Yes"
- Use "It depends" when coverage is conditional or unclear
- Keep the explanation clear and understandable
- Use inline emojis (✔️/❌/📌) within normal sentences, NOT bullet points or newlines

Respond with ONLY the JSON, nothing else."""
    
    def _extract_references(self, chunks: List[Dict]) -> List[str]:
        """Extract policy section references from chunks."""
        references = []
        seen_sections = set()
        
        for chunk in chunks[:5]:  # Top 5 most relevant
            section_ref = (
                f"{chunk['policy_type'].capitalize()} Policy - "
                f"{chunk['section_name']}"
            )
            
            if section_ref not in seen_sections:
                references.append(section_ref)
                seen_sections.add(section_ref)
        
        return references
    
    def _get_disclaimer(self) -> str:
        """Get standard disclaimer text."""
        return (
            "This information is for educational purposes only and does not "
            "constitute a coverage determination or claim decision. Actual coverage "
            "depends on the specific facts and circumstances of your situation and "
            "the complete terms and conditions of your policy. For official coverage "
            "determinations, please contact your insurance company or agent."
        )


if __name__ == "__main__":
    # Test the QA agent
    agent = QAAgent()
    
    # Mock retrieved chunks
    test_chunks = [
        {
            "text": "Coverage D2: Comprehensive Coverage\nWe will pay for direct and accidental loss to your covered auto from any cause except collision, including theft or larceny.",
            "section_name": "Part D - Physical Damage Coverage",
            "policy_type": "auto",
            "policy_file": "auto_policy_1.md",
            "chunk_type": "coverage",
            "similarity": 0.85
        },
        {
            "text": "Vehicle stolen from parking lot - Comprehensive Coverage applies (after $250 deductible)",
            "section_name": "Common Coverage Scenarios",
            "policy_type": "auto",
            "policy_file": "auto_policy_1.md",
            "chunk_type": "coverage",
            "similarity": 0.82
        }
    ]
    
    question = "Am I covered if my car is stolen?"
    
    response = agent.generate_response(
        question=question,
        retrieved_chunks=test_chunks,
        policy_types_queried=["auto"]
    )
    
    print("Q&A Agent Test\n")
    print(f"Question: {question}\n")
    print(f"Policy Applied: {response['policy_applied']}")
    print(f"Coverage Result: {response['coverage_result']}")
    print(f"\nExplanation: {response['explanation']}")
    print(f"\nPolicy References:")
    for ref in response['policy_references']:
        print(f"  - {ref}")
