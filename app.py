"""
Policy Understanding Simulator
A POC application for answering coverage questions using RAG and AI agents.
"""

import streamlit as st
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import our modules
from rag.policy_loader import PolicyLoader
from rag.embeddings import EmbeddingGenerator
from rag.retriever import PolicyRetriever
from agents.scenario_interpreter import ScenarioInterpreter
from agents.policy_selector import PolicySelector
from agents.explanation_agent import ExplanationAgent
from agents.retrieval_agent import PolicyRetrievalAgent
from agents.orchestrator import AgentOrchestrator



# Page configuration
st.set_page_config(
    page_title="Policy Understanding Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Headers */
    h1 {
        color: #1a202c;
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    h2 {
        color: #2d3748;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Card-like containers */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Coverage badges */
    .coverage-yes {
        background: #48bb78;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .coverage-no {
        background: #f56565;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .coverage-depends {
        background: #ed8936;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    .policy-badge {
        background: #4299e1;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.orchestrator = None
    st.session_state.loader = None
    st.session_state.current_user = None
    st.session_state.clarifying_question = None


@st.cache_resource
def initialize_system():
    """Initialize the RAG system and agents (cached)."""
    with st.spinner("🔄 Loading policy documents and initializing AI agents..."):
        # Initialize components
        embedding_gen = EmbeddingGenerator()
        retriever = PolicyRetriever(embedding_gen)
        loader = PolicyLoader("policies")
        
        # Load all 4 policies and build index
        all_chunks = []
        for policy_file in ["auto_policy_1.md", "auto_policy_2.md", 
                           "property_policy_1.md", "property_policy_2.md"]:
            chunks = loader.load_policy(policy_file)
            all_chunks.extend(chunks)
        retriever.build_index(all_chunks)
        
        # Initialize the 4 agents
        scenario_interpreter = ScenarioInterpreter()
        from agents.scenario_classifier import ScenarioClassifier
        classifier = ScenarioClassifier()
        policy_selector = PolicySelector(classifier)
        retrieval_agent = PolicyRetrievalAgent(retriever, loader)
        explanation_agent = ExplanationAgent()
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(
            policy_selector=policy_selector,
            scenario_interpreter=scenario_interpreter,
            retrieval_agent=retrieval_agent,
            explanation_agent=explanation_agent
        )
        
        return orchestrator, loader


def load_users():
    """Load user configuration."""
    with open("config/users.json", 'r') as f:
        return json.load(f)


def get_coverage_badge_html(result: str) -> str:
    """Generate HTML for coverage result badge."""
    if result.lower() == "yes":
        return '<span class="coverage-yes">✓ COVERED</span>'
    elif result.lower() == "no":
        return '<span class="coverage-no">✗ NOT COVERED</span>'
    else:
        return '<span class="coverage-depends">⚠ IT DEPENDS</span>'


def main():
    """Main application."""
    
    # Header
    st.title("🛡️ Policy Understanding Simulator")
    st.markdown("##### Ask questions about your insurance coverage")
    
    # Sidebar - User Selection
    st.sidebar.header("👤 Demo User Selection")
    
    users = load_users()
    user_options = {
        "alice": f"👩‍💼 Alice - Auto Only",
        "bob": f"👨‍💼 Bob - Property Only",
        "carol": f"👩‍💻 Carol - Auto & Property"
    }
    
    selected_user = st.sidebar.selectbox(
        "Select a demo user:",
        options=list(user_options.keys()),
        format_func=lambda x: user_options[x],
        key="user_selector"
    )
    
    st.session_state.current_user = selected_user
    user_data = users[selected_user]
    
    # Removed duplicate "Your Policies" section - same info shown on main screen
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Example Questions:**")
    st.sidebar.caption("• Am I covered if my car is stolen?")
    st.sidebar.caption("• What if my house catches fire?")
    st.sidebar.caption("• Is flood damage covered?")
    st.sidebar.caption("• Am I covered for hail damage?")
    
    # Initialize system
    if not st.session_state.initialized:
        (st.session_state.orchestrator,
         st.session_state.loader) = initialize_system()
        st.session_state.initialized = True
    
    # Main content area
    st.markdown("---")
    
    # Display current user's policies prominently
    st.markdown("### 📋 Your Current Policies")
    policy_cols = st.columns(len(user_data["policies"]))
    
    for idx, (policy_type, policy_file) in enumerate(user_data["policies"].items()):
        with policy_cols[idx]:
            if policy_type == "auto":
                if selected_user == "alice":
                    st.info("🚗 **Auto Insurance**\nStandard Coverage")
                elif selected_user == "carol":
                    st.info("🚗 **Auto Insurance**\nPremium Coverage")
            elif policy_type == "property":
                if selected_user == "bob":
                    st.success("🏠 **Property Insurance**\nBasic Coverage")
                elif selected_user == "carol":
                    st.success("🏠 **Property Insurance**\nComprehensive Coverage")
    
    st.markdown("---")
    
    # Show clarifying question above text area if present
    if st.session_state.clarifying_question:
        st.warning(f"⚠️ {st.session_state.clarifying_question}")
    
    # Question input
    question = st.text_area(
        "🔍 Ask your coverage question:",
        placeholder="Example: Am I covered if my car is damaged in a flood?",
        height=100,
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1.2, 0.6, 3.2])
    with col1:
        submit_button = st.button("🔍 Analyze Coverage", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.clarifying_question = None
        st.rerun()
    
    # Process question
    if submit_button and question.strip():
        with st.spinner("🤔 Running 4-agent workflow: Policy Selection → Scenario Interpretation → Retrieval → Explanation..."):
            try:
                # Get user's policies
                user_policies = users[selected_user]["policies"]
                
                # Use orchestrator to process question through 4-agent workflow
                response = st.session_state.orchestrator.process_question(
                    question=question,
                    user_policies=user_policies,
                    selected_user=selected_user
                )
                
                # Handle clarification case
                if response.get("needs_clarification"):
                    st.session_state.clarifying_question = response.get("clarification_question")
                    st.rerun()
                else:
                    # Clear any previous clarifying question
                    st.session_state.clarifying_question = None
                    
                    # Display response in specified format
                    st.markdown("---")
                    st.markdown("### 📊 Coverage Analysis Result")
                    
                    # Response format matching specification
                    st.markdown(f"**Selected User:** {response['selected_user'].upper()}")
                    st.markdown(f"**Policy Applied:** {response['policy_applied']}")
                    st.markdown("")
                    
                    # Coverage Answer with badge
                    st.markdown("**Coverage Answer:**")
                    st.markdown(get_coverage_badge_html(response["coverage_result"]), unsafe_allow_html=True)
                    st.markdown("")
                    
                    # Explanation - Clean formatting without emojis
                    st.markdown("**Explanation:**")
                    explanation_text = response["explanation"]
                    
                    # Remove emojis and symbols
                    import re
                    # Remove common emojis
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001F900-\U0001F9FF"  # supplemental symbols
                        "]+", flags=re.UNICODE)
                    explanation_text = emoji_pattern.sub('', explanation_text)
                    
                    # Clean up any leftover symbols and extra spaces
                    explanation_text = explanation_text.replace('- ', '')
                    explanation_text = explanation_text.replace('• ', '')
                    explanation_text = re.sub(r'\s+', ' ', explanation_text)  # Multiple spaces to single
                    explanation_text = explanation_text.strip()
                    
                    st.info(explanation_text)
                    
                    # Policy References
                    st.markdown("**Policy References:**")
                    for ref in response["policy_references"]:
                        st.markdown(f"- {ref}")
                    
                    st.markdown("")
                    
                    # Disclaimer
                    st.markdown("**Disclaimer:**")
                    st.caption(response["disclaimer"])
                    
                    # Additional details in expanders
                    st.markdown("---")
                    
                    # Show scenario interpretation
                    with st.expander("🔍 Scenario Interpretation (Agent 2)", expanded=False):
                        scenario = response.get("scenario_details", {})
                        st.markdown(f"**Asset:** {scenario.get('asset', 'N/A')}")
                        st.markdown(f"**Event:** {scenario.get('event', 'N/A')}")
                        st.markdown(f"**Location:** {scenario.get('location', 'N/A')}")
                        st.markdown(f"**Reasoning:** {scenario.get('reasoning', 'N/A')}")
                    
                    # Show agent workflow trace
                    with st.expander("🤖 Agent Workflow Trace", expanded=False):
                        for trace_line in response.get("agent_trace", []):
                            st.caption(trace_line)
                    
                    # Show retrieved chunks (for transparency)
                    with st.expander("📄 Retrieved Policy Text", expanded=False):
                        for i, chunk in enumerate(response.get("retrieved_chunks", [])[:3], 1):
                            st.markdown(f"**{i}. {chunk['section_name']}** "
                                      f"(Relevance: {chunk['similarity']:.0%})")
                            st.caption(chunk['text'][:300] + "...")
                            st.markdown("---")
                    
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")
                st.caption(f"Error details: {str(e)}")
                import traceback
                st.caption(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    


if __name__ == "__main__":
    main()
