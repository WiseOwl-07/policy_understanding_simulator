"""
Scenario classifier agent.
Determines if a user question is Auto-specific, Property-specific, or ambiguous.
"""

from typing import Dict
from groq import Groq
import os


class ScenarioClassifier:
    """Classifies user scenarios into Auto, Property, or Ambiguous categories."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the classifier.
        
        Args:
            api_key: Groq API key (or will use GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"  # Fast and accurate
    
    def classify(self, question: str) -> Dict[str, str]:
        """
        Classify the user's question.
        
        Args:
            question: User's coverage question
        
        Returns:
            Dictionary with:
                - classification: "auto", "property", or "ambiguous"
                - confidence: "high", "medium", or "low"
                - reasoning: Explanation of classification
        """
        prompt = f"""You are an insurance policy classifier. Analyze the following coverage question and determine if it relates to AUTO insurance, PROPERTY (home) insurance, or if it's AMBIGUOUS (could be either).

Question: "{question}"

AUTO insurance typically covers:
- Vehicles, cars, motorcycles
- Vehicle theft, damage, collisions
- Auto accidents, traffic incidents
- Vehicle comprehensive/collision coverage

PROPERTY insurance typically covers:
- Houses, homes, residences, dwellings
- Home damage (fire, wind, theft inside home)
- Property structures (roof, walls, foundation)
- Personal belongings inside home

AMBIGUOUS scenarios might include:
- Generic terms like "fire" or "theft" without specifying what/where
- "Water damage" without context
- Scenarios that could affect both (e.g., "garage" could be part of home OR where car is parked)

Respond in this exact JSON format:
{{
    "classification": "auto" OR "property" OR "ambiguous",
    "confidence": "high" OR "medium" OR "low",
    "reasoning": "Brief explanation of why this classification was chosen"
}}

Only respond with the JSON, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful insurance classification assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        import json
        try:
            # Remove markdown code blocks if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {result_text}")
            # Return default
            return {
                "classification": "ambiguous",
                "confidence": "low",
                "reasoning": "Unable to parse classification response"
            }


if __name__ == "__main__":
    # Test the classifier
    classifier = ScenarioClassifier()
    
    test_questions = [
        "Am I covered if my car is stolen?",
        "What if my house catches fire?",
        "Is flood damage covered?",  # Ambiguous
        "My vehicle was damaged in an accident",
        "Someone broke into my home",
        "What about fire damage?",  # Ambiguous
    ]
    
    print("Testing Scenario Classifier\n")
    for question in test_questions:
        result = classifier.classify(question)
        print(f"Q: {question}")
        print(f"   Classification: {result['classification'].upper()}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Reasoning: {result['reasoning']}\n")
