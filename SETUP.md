# Setup Instructions

## Quick Start

1. **Set your Groq API key** in the `.env` file:
   ```bash
   echo "GROQ_API_KEY=your-actual-groq-api-key" > .env
   ```

   Or export it:
   ```bash
   export GROQ_API_KEY='your-actual-groq-api-key'
   ```

2. **Install dependencies** (if not already done):
   ```bash
   pip3 install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**: The app will open automatically, or navigate to `http://localhost:8501`

## Testing Scenarios

### Auto Insurance (Alice)
- "Am I covered if my car is stolen?"
- "What if I hit another car?"
- "Is flood damage to my car covered?"

### Property Insurance (Bob)
- "Am I covered if my house catches fire?"
- "What about theft from my home?"
- "Is earthquake damage covered?"

### Both Policies (Carol)
- "My car was in the garage when fire destroyed my house. Am I covered?"
- "Is flood damage covered?" (should trigger clarification)

---

For full documentation, see [README.md](README.md)
