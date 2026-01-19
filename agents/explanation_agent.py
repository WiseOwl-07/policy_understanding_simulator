"""
Agent 4: Explanation Agent
Generates plain-English explanations based on retrieved policy chunks.
Does NOT make claim decisions, clearly states assumptions.
"""

from typing import List, Dict
from groq import Groq
import os
import json


class ExplanationAgent:
    """Agent 4: Generates structured coverage responses from policy chunks with plain-English explanations."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the QA agent.
        
        Args:
            api_key: Groq API key (or will use GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "deepseek-r1-distill-llama-70b"
    
    def generate_response(self, question: str, retrieved_chunks: List[Dict],
                         policy_types_queried: List[str], scenario_details: Dict = None) -> Dict:
        """
        Generate a structured coverage response.
        
        Args:
            question: User's coverage question
            retrieved_chunks: List of retrieved policy chunks with metadata
            policy_types_queried: List of policy types that were queried
            scenario_details: Optional structured scenario details from Agent 2
        
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
2. Provide a clear explanation in a semi-formal friendly tone - professional yet approachable
3. Write in plain language that's easy to understand, using some contractions (doesn't, you'll, etc.) where natural
4. Keep the explanation concise (2-4 complete sentences)
5. Reference specific policy sections when relevant (e.g., "as stated in Part D - Physical Damage Coverage")
6. Use phrases like "I've reviewed", "we're looking at", "unfortunately" to sound helpful and empathetic
7. DO NOT use any emojis, symbols, or bullet points in the explanation
8. Write the explanation as flowing paragraph text that sounds like a helpful insurance agent explaining things

Respond in this exact JSON format:
{{
    "coverage_result": "Yes" OR "No" OR "It depends",
    "explanation": "A clear, semi-formal friendly explanation. Example: I've reviewed your auto insurance policy, and unfortunately it doesn't provide coverage for a house fire. The policy excerpts we're looking at focus specifically on vehicle-related coverage - sections like Part D Physical Damage Coverage address damage to your car, not homeowner's insurance. While your policy does include Comprehensive Coverage for fires and explosions, this applies only to your vehicle, not your home. To protect your house against fire damage, you would need a separate homeowner's insurance policy."
}}

IMPORTANT: 
- Base your answer ONLY on the provided policy excerpts
- Do not make assumptions about coverage not mentioned in the excerpts
- If the policy explicitly excludes something, say "No"
- If the policy explicitly covers something, say "Yes"
- Use "It depends" when coverage is conditional or unclear
- Maintain a semi-formal friendly tone - professional but warm and approachable
- Use some contractions naturally (doesn't, you'll, won't) but maintain professionalism
- NO emojis, NO symbols, NO bullet points - write as a flowing paragraph

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
