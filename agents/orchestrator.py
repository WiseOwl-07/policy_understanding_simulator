"""
Agent Orchestrator
Coordinates the 4-agent workflow for policy understanding:
1. Policy Selection Agent → 2. Scenario Interpreter → 3. Retrieval Agent → 4. Explanation Agent
"""

from typing import Dict, List, Optional
from agents.policy_selector import PolicySelector
from agents.scenario_interpreter import ScenarioInterpreter
from agents.retrieval_agent import PolicyRetrievalAgent
from agents.explanation_agent import ExplanationAgent


class AgentOrchestrator:
    """
    Orchestrates the multi-agent workflow for policy understanding.
    
    Workflow:
    1. Agent 1 (Policy Selection): Determine which policies to query
    2. Agent 2 (Scenario Interpretation): Extract structured scenario details
    3. Agent 3 (Policy Retrieval): Retrieve relevant policy clauses via RAG
    4. Agent 4 (Explanation): Generate plain-English explanation
    """
    
    def __init__(
        self,
        policy_selector: PolicySelector,
        scenario_interpreter: ScenarioInterpreter,
        retrieval_agent: PolicyRetrievalAgent,
        explanation_agent: ExplanationAgent
    ):
        """
        Initialize the orchestrator with all 4 agents.
        
        Args:
            policy_selector: Agent 1 - Policy Selection Agent
            scenario_interpreter: Agent 2 - Scenario Interpretation Agent
            retrieval_agent: Agent 3 - Policy Retrieval Agent (RAG)
            explanation_agent: Agent 4 - Explanation Agent
        """
        self.policy_selector = policy_selector
        self.scenario_interpreter = scenario_interpreter
        self.retrieval_agent = retrieval_agent
        self.explanation_agent = explanation_agent
    
    def process_question(
        self,
        question: str,
        user_policies: Dict[str, str],
        selected_user: str
    ) -> Dict:
        """
        Process a user's coverage question through the 4-agent workflow.
        
        Args:
            question: User's coverage question
            user_policies: Dictionary of user's policies {type: filename}
            selected_user: User identifier (e.g., "alice", "bob", "carol")
        
        Returns:
            Dictionary with:
                - selected_user: User identifier
                - policy_applied: "Auto" / "Property" / "Both"
                - coverage_result: "Yes" / "No" / "It depends"
                - explanation: Plain-English explanation
                - policy_references: List of policy sections cited
                - disclaimer: Standard disclaimer
                - needs_clarification: Boolean
                - clarification_question: Optional clarification question
                - scenario_details: Structured scenario interpretation
                - agent_trace: Debugging information about agent workflow
        """
        agent_trace = []
        
        # Step 1: Scenario Interpretation (Agent 2)
        # Note: We run interpretation first to get structured details
        agent_trace.append("Agent 2: Interpreting scenario...")
        scenario_details = self.scenario_interpreter.interpret(question)
        agent_trace.append(f"  → Asset: {scenario_details['asset']}, Event: {scenario_details['event']}, Location: {scenario_details['location']}")
        
        # Step 2: Policy Selection (Agent 1)
        # Uses scenario interpretation to select appropriate policies
        agent_trace.append("Agent 1: Selecting policies...")
        selection_result = self.policy_selector.select_policies(question, user_policies)
        
        policies_to_query = selection_result["policies_to_query"]
        needs_clarification = selection_result["needs_clarification"]
        
        agent_trace.append(f"  → Policies to query: {policies_to_query}")
        
        # If clarification is needed, return early
        if needs_clarification:
            agent_trace.append("  → Clarification needed, returning...")
            return {
                "selected_user": selected_user,
                "policy_applied": "Unknown",
                "coverage_result": "It depends",
                "explanation": "Please clarify your question to get a specific coverage answer.",
                "policy_references": [],
                "disclaimer": self.explanation_agent._get_disclaimer(),
                "needs_clarification": True,
                "clarification_question": selection_result.get("clarification_question"),
                "scenario_details": scenario_details,
                "agent_trace": agent_trace
            }
        
        # Step 3: Policy Retrieval (Agent 3)
        agent_trace.append("Agent 3: Retrieving relevant policy clauses...")
        
        # Retrieve from demo policies
        retrieved_chunks = self.retrieval_agent.retrieve_relevant_clauses(
            question=question,
            scenario_details=scenario_details,
            user_policies=user_policies,
            policy_types_to_query=policies_to_query,
            top_k=5
        )
        
        agent_trace.append(f"  → Retrieved {len(retrieved_chunks)} chunks from demo policies")
        
        if not retrieved_chunks:
            agent_trace.append("  → No relevant chunks found!")
            return {
                "selected_user": selected_user,
                "policy_applied": "Unknown",
                "coverage_result": "It depends",
                "explanation": "Could not find relevant policy information to answer your question. Please try rephrasing.",
                "policy_references": [],
                "disclaimer": self.explanation_agent._get_disclaimer(),
                "needs_clarification": False,
                "clarification_question": None,
                "scenario_details": scenario_details,
                "agent_trace": agent_trace
            }
        
        # Step 4: Generate Explanation (Agent 4)
        agent_trace.append("Agent 4: Generating explanation...")
        response = self.explanation_agent.generate_response(
            question=question,
            retrieved_chunks=retrieved_chunks,
            policy_types_queried=policies_to_query,
            scenario_details=scenario_details
        )
        
        agent_trace.append("  → Explanation generated")
        
        # Add metadata to response
        response["selected_user"] = selected_user
        response["needs_clarification"] = False
        response["clarification_question"] = None
        response["scenario_details"] = scenario_details
        response["agent_trace"] = agent_trace
        response["retrieved_chunks"] = retrieved_chunks  # For transparency
        
        return response
    
    def get_agent_status(self) -> Dict:
        """
        Get status of all agents.
        
        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_1": "Policy Selection Agent - Ready",
            "agent_2": "Scenario Interpretation Agent - Ready",
            "agent_3": "Policy Retrieval Agent - Ready",
            "agent_4": "Explanation Agent - Ready",
            "orchestrator": "Ready"
        }


if __name__ == "__main__":
    print("Agent Orchestrator")
    print("Coordinates the 4-agent workflow:")
    print("  1. Policy Selection → 2. Scenario Interpretation → 3. Retrieval → 4. Explanation")
    print("\nRun the full application to test the orchestration.")
