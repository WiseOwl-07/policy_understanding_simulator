"""
Policy selector agent.
Determines which policy/policies to query based on user's policies and scenario classification.
"""

from typing import Dict, List, Optional
from agents.scenario_classifier import ScenarioClassifier


class PolicySelector:
    """Selects appropriate policies to query based on user and scenario."""
    
    def __init__(self, scenario_classifier: ScenarioClassifier):
        """
        Initialize the policy selector.
        
        Args:
            scenario_classifier: ScenarioClassifier instance
        """
        self.classifier = scenario_classifier
    
    def select_policies(self, question: str, user_policies: Dict[str, str]) -> Dict:
        """
        Determine which policies to query for this question.
        
        Args:
            question: User's coverage question
            user_policies: Dictionary of user's policies 
                          e.g., {"auto": "auto_policy_1.md"} or 
                                {"auto": "...", "property": "..."}
        
        Returns:
            Dictionary with:
                - policies_to_query: List of policy types to search (e.g., ["auto"])
                - needs_clarification: Boolean
                - clarification_question: Optional string with question for user
                - classification: The scenario classification result
        """
        available_types = list(user_policies.keys())
        
        # If user has only one policy type, always use it
        if len(available_types) == 1:
            return {
                "policies_to_query": available_types,
                "needs_clarification": False,
                "clarification_question": None,
                "classification": {
                    "classification": available_types[0],
                    "confidence": "high",
                    "reasoning": f"User only has {available_types[0]} insurance"
                }
            }
        
        # User has multiple policy types - need to classify
        classification = self.classifier.classify(question)
        
        scenario_type = classification["classification"]
        confidence = classification["confidence"]
        
        # Clear classification for auto or property
        if scenario_type == "auto":
            if "auto" in available_types:
                return {
                    "policies_to_query": ["auto"],
                    "needs_clarification": False,
                    "clarification_question": None,
                    "classification": classification
                }
            else:
                # User doesn't have auto insurance but question is about auto
                return {
                    "policies_to_query": available_types,  # Query what they have
                    "needs_clarification": False,
                    "clarification_question": None,
                    "classification": classification
                }
        
        elif scenario_type == "property":
            if "property" in available_types:
                return {
                    "policies_to_query": ["property"],
                    "needs_clarification": False,
                    "clarification_question": None,
                    "classification": classification
                }
            else:
                # User doesn't have property insurance but question is about property
                return {
                    "policies_to_query": available_types,  # Query what they have
                    "needs_clarification": False,
                    "clarification_question": None,
                    "classification": classification
                }
        
        # User explicitly asked about all policies (e.g., "any of my policies")
        elif scenario_type == "both":
            return {
                "policies_to_query": available_types,  # Query all available policies
                "needs_clarification": False,
                "clarification_question": None,
                "classification": classification
            }
        
        # Ambiguous classification - ask for clarification
        else:
            clarification_q = self._generate_clarification_question(
                question, available_types
            )
            
            # For now, query all available policies
            # In a real interactive system, would wait for user clarification
            return {
                "policies_to_query": available_types,
                "needs_clarification": True,
                "clarification_question": clarification_q,
                "classification": classification
            }
    
    def _generate_clarification_question(self, question: str, 
                                        available_types: List[str]) -> str:
        """Generate a clarification question for ambiguous scenarios."""
        type_str = " or ".join([t.capitalize() for t in available_types])
        
        return (f"Your question could relate to either {type_str} insurance. "
                f"Are you asking about your vehicle or your home/property?")


if __name__ == "__main__":
    # Test the policy selector
    classifier = ScenarioClassifier()
    selector = PolicySelector(classifier)
    
    # Test with user who has both policies
    carol_policies = {
        "auto": "auto_policy_2.md",
        "property": "property_policy_2.md"
    }
    
    test_questions = [
        "Am I covered if my car is stolen?",  # Clear auto
        "What if my house catches fire?",  # Clear property
        "Is flood damage covered?",  # Ambiguous
    ]
    
    print("Testing Policy Selector (User: Carol - has both policies)\n")
    for question in test_questions:
        result = selector.select_policies(question, carol_policies)
        print(f"Q: {question}")
        print(f"   Policies to query: {result['policies_to_query']}")
        print(f"   Needs clarification: {result['needs_clarification']}")
        if result['needs_clarification']:
            print(f"   Clarification: {result['clarification_question']}")
        print(f"   Classification: {result['classification']['classification']} "
              f"({result['classification']['confidence']} confidence)")
        print()
