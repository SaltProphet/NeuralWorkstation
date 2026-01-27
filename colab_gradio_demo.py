"""
Colab Gradio Demo for Autonomous Agent System

This file provides a Colab-ready demo with a two-column chat interface.
Copy and paste the code below into a Google Colab cell to run the demo.
"""

COLAB_CELL_CODE = '''
# ========================================================================
# AUTONOMOUS AGENT SYSTEM - COLAB GRADIO DEMO
# ========================================================================
# 
# This cell installs dependencies, sets up the agent system, and launches
# a Gradio interface with chat UI and reasoning log visualization.
# 
# Usage:
# 1. Copy this entire cell
# 2. Paste into a Google Colab notebook
# 3. Run the cell
# 4. Set your GROQ_API_KEY in the Colab environment (optional)
# 5. Use the shareable link to demo the agent system
# ========================================================================

# Step 1: Install dependencies
print("📦 Installing dependencies...")
!pip install -q groq qdrant-client sentence-transformers gradio python-dotenv

# Step 2: Import required modules
print("📚 Importing modules...")
import os
import gradio as gr
from typing import List, Tuple

# Step 3: Set up agent system code (inline for Colab portability)
print("🔧 Setting up agent system...")

# ========================================================================
# Agent Roles Module (inline)
# ========================================================================

AGENT_ROLES = {
    'gatekeeper': {
        'name': 'Gatekeeper',
        'system_prompt': """You are a Gatekeeper agent that classifies user intents.

Your task is to analyze the user's message and determine their intent. Respond with a JSON object containing:
- "intent": one of ["sales", "support", "general", "greeting"]
- "confidence": a number between 0.0 and 1.0
- "reason": a brief explanation for your classification

Be concise and accurate. Always respond with valid JSON only."""
    },
    'closer': {
        'name': 'Closer',
        'system_prompt': """You are a Closer agent focused on sales and conversions.

Your goal is to help customers make informed purchasing decisions.
Keep messages concise (under 100 words) and action-oriented.
Use friendly, professional language.
Always include a clear call-to-action."""
    }
}

def get_system_prompt(role_key: str) -> str:
    """Get system prompt for a role"""
    return AGENT_ROLES.get(role_key, {}).get('system_prompt', f"You are a {role_key} agent.")

# ========================================================================
# LLM Client and Agent System (inline)
# ========================================================================

import json
import hashlib
from datetime import datetime

class LLMClient:
    """Groq LLM Client with fallback to stubs"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except:
                pass
    
    def is_available(self):
        return self.client is not None
    
    def chat_completion(self, messages, temperature=0.7):
        if not self.is_available():
            return self._stub_response(messages)
        
        try:
            response = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except:
            return self._stub_response(messages)
    
    def _stub_response(self, messages):
        user_msg = ""
        for msg in messages:
            if msg.get('role') == 'user':
                user_msg = msg.get('content', '').lower()
        
        if 'price' in user_msg or 'cost' in user_msg or 'buy' in user_msg:
            return '{"intent": "sales", "confidence": 0.8, "reason": "Pricing query (stub)"}'
        elif 'hello' in user_msg or 'hi' in user_msg:
            return '{"intent": "greeting", "confidence": 0.9, "reason": "Greeting (stub)"}'
        elif 'help' in user_msg or 'support' in user_msg:
            return '{"intent": "support", "confidence": 0.8, "reason": "Support request (stub)"}'
        else:
            return '{"intent": "general", "confidence": 0.6, "reason": "General query (stub)"}'

class MemoryManager:
    """Simple in-memory storage"""
    
    def __init__(self):
        self.store = []
    
    def store_item(self, text, metadata=None):
        self.store.append({
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def search(self, query, limit=5):
        results = []
        query_lower = query.lower()
        for item in self.store:
            if query_lower in item['text'].lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results

# Agent Workflow
def autonomous_agent_main(user_input: str):
    """Main agent workflow"""
    llm_client = LLMClient()
    memory = MemoryManager()
    
    price_data = {
        "basic": {"price": "$29/month", "features": ["Core features", "Email support"]},
        "pro": {"price": "$99/month", "features": ["All features", "Priority support"]},
        "enterprise": {"price": "Custom", "features": ["Custom solutions", "Dedicated support"]}
    }
    
    reasoning_log = []
    
    # Step 1: Intent classification
    reasoning_log.append("🔍 Step 1: Classifying intent...")
    system_prompt = get_system_prompt('gatekeeper')
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    intent_response = llm_client.chat_completion(messages, temperature=0.3)
    
    try:
        intent_result = json.loads(intent_response)
    except:
        intent_result = {
            "intent": "general",
            "confidence": 0.5,
            "reason": "Parse error"
        }
    
    intent = intent_result.get('intent', 'general')
    confidence = intent_result.get('confidence', 0.5)
    
    reasoning_log.append(f"   Intent: {intent} (confidence: {confidence:.2f})")
    reasoning_log.append(f"   Reason: {intent_result.get('reason', 'N/A')}")
    
    # Step 2: Generate response
    reasoning_log.append(f"\\n📍 Step 2: Routing to {intent} handler...")
    
    if intent == 'sales':
        closer_prompt = get_system_prompt('closer')
        user_msg = f"""User inquiry: {user_input}

Available pricing:
{json.dumps(price_data, indent=2)}

Generate a concise sales message."""
        
        messages = [
            {"role": "system", "content": closer_prompt},
            {"role": "user", "content": user_msg}
        ]
        response = llm_client.chat_completion(messages, temperature=0.7)
        reasoning_log.append("   ✓ Sales closer generated response")
    
    elif intent == 'support':
        response = ("Thank you for reaching out! I've recorded your support request. "
                   "A team member will assist you shortly. How can I help you right now?")
        reasoning_log.append("   ✓ Support handler processed request")
    
    elif intent == 'greeting':
        response = ("Hello! 👋 Welcome! I'm your AI assistant. I can help you with:\\n"
                   "- Product pricing and features\\n"
                   "- Technical support\\n"
                   "- General questions\\n\\n"
                   "What can I do for you today?")
        reasoning_log.append("   ✓ Greeting handler responded")
    
    else:
        response = ("Thanks for your message! I can help you with:\\n"
                   "- Product pricing and features (sales)\\n"
                   "- Technical support (support)\\n"
                   "- General information\\n\\n"
                   "How can I assist you?")
        reasoning_log.append("   ✓ General handler provided info")
    
    reasoning_log.append("\\n💾 Step 3: Storing interaction...")
    memory.store_item(f"User: {user_input}\\nIntent: {intent}\\nResponse: {response[:50]}...")
    reasoning_log.append("   ✓ Interaction stored")
    
    return {
        "response": response,
        "reasoning_log": "\\n".join(reasoning_log),
        "llm_available": llm_client.is_available()
    }

# ========================================================================
# Gradio Interface
# ========================================================================

print("🎨 Building Gradio interface...")

def process_message(message: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    """
    Process user message through agent system.
    
    Args:
        message: User's input message
        history: Chat history (not used in this simple version)
    
    Returns:
        Tuple of (response, reasoning_log)
    """
    if not message.strip():
        return "Please enter a message.", "No input provided."
    
    result = autonomous_agent_main(message)
    return result["response"], result["reasoning_log"]

# Create Gradio interface
with gr.Blocks(title="Autonomous Agent Demo") as demo:
    gr.Markdown("# 🤖 Autonomous Agent System - Interactive Demo")
    gr.Markdown("Ask questions about pricing, request support, or just say hello!")
    
    # Check LLM status
    llm_test = LLMClient()
    if llm_test.is_available():
        gr.Markdown("✅ **LLM Status:** Connected to Groq API")
    else:
        gr.Markdown("⚠️ **LLM Status:** Using deterministic stubs (set GROQ_API_KEY to enable)")
    
    with gr.Row():
        # Left column: Chat interface
        with gr.Column(scale=1):
            gr.Markdown("### 💬 Chat")
            chatbot = gr.Chatbot(label="Conversation", height=400)
            msg_input = gr.Textbox(
                label="Your Message",
                placeholder="Type your message here...",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
        
        # Right column: Reasoning log
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Reasoning Log")
            reasoning_output = gr.Textbox(
                label="Agent Reasoning",
                lines=20,
                max_lines=25,
                interactive=False
            )
    
    gr.Markdown("""
    ### 📝 Example Prompts
    - "How much does your pro plan cost?"
    - "I need help with my account"
    - "Hello! What can you do?"
    - "Tell me about your services"
    """)
    
    # Define interaction logic
    def respond(message, chat_history):
        """Handle user message and update UI"""
        if not message.strip():
            return chat_history, ""
        
        # Process message through agent
        agent_response, reasoning_log = process_message(message, chat_history)
        
        # Update chat history
        chat_history.append((message, agent_response))
        
        return chat_history, reasoning_log
    
    def clear_chat():
        """Clear chat history and reasoning log"""
        return [], ""
    
    # Wire up event handlers
    submit_btn.click(
        respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, reasoning_output]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    msg_input.submit(
        respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, reasoning_output]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    clear_btn.click(clear_chat, outputs=[chatbot, reasoning_output])

# ========================================================================
# Launch Demo
# ========================================================================

print("\\n" + "="*60)
print("🚀 Launching Gradio demo...")
print("="*60)
print()

# Launch with share=True to get a public link
demo.launch(share=True, debug=True)
'''


def print_colab_instructions():
    """
    Print instructions for using the Colab demo.
    """
    print("=" * 70)
    print("COLAB GRADIO DEMO - SETUP INSTRUCTIONS")
    print("=" * 70)
    print()
    print("To run the Autonomous Agent demo in Google Colab:")
    print()
    print("1. Open Google Colab: https://colab.research.google.com/")
    print("2. Create a new notebook")
    print("3. Copy the code below into a cell")
    print("4. (Optional) Set your API key in the cell:")
    print("   import os")
    print('   os.environ["GROQ_API_KEY"] = "your_api_key_here"')
    print("5. Run the cell")
    print("6. Click the generated Gradio share link to access the demo")
    print()
    print("=" * 70)
    print()


def get_colab_code() -> str:
    """
    Get the complete Colab cell code.
    
    Returns:
        String containing the complete Colab-ready code
    """
    return COLAB_CELL_CODE


def save_colab_code(filepath: str = "colab_demo_cell.py"):
    """
    Save the Colab code to a file for easy copying.
    
    Args:
        filepath: Path where to save the code
    """
    with open(filepath, 'w') as f:
        f.write(COLAB_CELL_CODE)
    print(f"✓ Colab demo code saved to: {filepath}")


if __name__ == "__main__":
    print_colab_instructions()
    print("=" * 70)
    print("COLAB CELL CODE (Copy everything below)")
    print("=" * 70)
    print()
    print(COLAB_CELL_CODE)
    print()
    print("=" * 70)
    print("End of Colab cell code")
    print("=" * 70)
