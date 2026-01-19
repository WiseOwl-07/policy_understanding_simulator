"""
Agent 2: Scenario Interpretation Agent
Extracts structured details from user questions: Asset, Event, Location
"""

from typing import Dict
from groq import Groq
import os
import json


class ScenarioInterpreter:
    """
    Agent 2: Interprets user scenarios and extracts structured information.
    
    Extracts:
    - Asset: What is being insured (car, house, contents, vehicle, roof, etc.)
    - Event: What happened (flood, fire, theft, accident, hail, collision, etc.)
    - Location: Where it occurred (road, garage, inside house, driveway, parked, etc.)
    - Policy Type: Auto, Property, or Both (for routing)
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the scenario interpreter.
        
        Args:
            api_key: Groq API key (or will use GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
    
    def interpret(self, question: str) -> Dict:
        """
        Extract structured information from the user's question.
        
        Args:
            question: User's coverage question
        
        Returns:
            Dictionary with:
                - asset: What is being insured (e.g., "car", "house", "vehicle")
                - event: What happened (e.g., "flood", "fire", "theft")
                - location: Where it occurred (e.g., "road", "garage", "home")
                - policy_type: "auto", "property", or "ambiguous"
                - confidence: "high", "medium", or "low"
                - reasoning: Explanation of interpretation
                - needs_clarification: Boolean
        """
        prompt = f"""You are an insurance scenario interpreter. Extract structured information from the user's coverage question.

Question: "{question}"

Extract the following information:

1. ASSET - What is being insured?
   Examples: car, vehicle, house, home, roof, contents, personal property, dwelling

2. EVENT - What happened or what is the user asking about?
   Examples: theft, fire, flood, collision, accident, hail, wind, water damage, break-in

3. LOCATION - Where did it occur or where is the context?
   Examples: road, highway, garage, driveway, parked, inside house, outside, at home

4. POLICY_TYPE - Which insurance type applies?
   - "auto" if clearly about a vehicle/car
   - "property" if clearly about a house/home/dwelling
   - "ambiguous" if unclear or could apply to both

5. CONFIDENCE - How confident are you in this interpretation?
   - "high" if all details are clear
   - "medium" if some details are inferred
   - "low" if the question is vague

6. NEEDS_CLARIFICATION - Does the user need to clarify?
   - true if policy_type is "ambiguous" AND confidence is "low" or "medium"
   - false otherwise

Respond in this exact JSON format:
{{
    "asset": "extracted asset",
    "event": "extracted event",
    "location": "extracted location or 'unknown'",
    "policy_type": "auto" OR "property" OR "ambiguous",
    "confidence": "high" OR "medium" OR "low",
    "reasoning": "Brief explanation of your interpretation",
    "needs_clarification": true OR false
}}

Only respond with the JSON, nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful insurance scenario interpretation assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400
        )
        
        result_text = response.choices[0].message.content.strip()
        
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
                "asset": "unknown",
                "event": "unknown",
                "location": "unknown",
                "policy_type": "ambiguous",
                "confidence": "low",
                "reasoning": "Unable to parse interpretation response",
                "needs_clarification": True
            }


if __name__ == "__main__":
    # Test the interpreter
    interpreter = ScenarioInterpreter()
    
    test_questions = [
        "Am I covered if my car is stolen?",
        "What if my house catches fire?",
        "Is flood damage covered?",  # Ambiguous
        "My car was flooded while parked in my garage",
        "Items stolen from my car parked at home",
        "Fire damaged my house",
        "Hail damaged my roof and my car"
    ]
    
    print("Testing Scenario Interpreter (Agent 2)\n")
    print("=" * 80 + "\n")
    
    for question in test_questions:
        result = interpreter.interpret(question)
        print(f"Question: {question}")
        print(f"  Asset: {result['asset']}")
        print(f"  Event: {result['event']}")
        print(f"  Location: {result['location']}")
        print(f"  Policy Type: {result['policy_type'].upper()}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Needs Clarification: {result['needs_clarification']}")
        print(f"  Reasoning: {result['reasoning']}")
        print("-" * 80 + "\n")
