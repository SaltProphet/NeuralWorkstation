"""
Gemini Agentic System - Autonomous agent workflow with LLM integration

This module provides an agentic system with intent routing, knowledge retrieval,
and sales closing capabilities. It integrates with Groq LLM and Qdrant vector database.
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime

# Attempt to import agent_roles, fall back to minimal prompts if not available
try:
    from agent_roles import get_system_prompt
    AGENT_ROLES_AVAILABLE = True
except ImportError:
    AGENT_ROLES_AVAILABLE = False
    # Fallback minimal prompts
    MINIMAL_PROMPTS = {
        'gatekeeper': 'You are a Gatekeeper. Classify the user intent as JSON: {"intent": "sales|support|general|greeting", "confidence": 0.0-1.0, "reason": "brief explanation"}',
        'closer': 'You are a Closer. Provide a concise sales message with pricing and call-to-action (under 100 words).'
    }
    
    def get_system_prompt(role_key: str) -> str:
        """Fallback function if agent_roles module is not available"""
        return MINIMAL_PROMPTS.get(role_key, f"You are a helpful {role_key} agent.")


# LLM Client - Handles Groq API interactions
class LLMClient:
    """
    Client for interacting with Groq LLM API.
    Falls back to deterministic stubs if API is unavailable.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mixtral-8x7b-32768"):
        """
        Initialize LLM client.
        
        Args:
            api_key: Groq API key (reads from GROQ_API_KEY env var if not provided)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.model = model
        self.client = None
        
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                print("[LLMClient] Warning: groq package not installed. Using stub mode.")
            except Exception as e:
                print(f"[LLMClient] Warning: Failed to initialize Groq client: {e}. Using stub mode.")
    
    def is_available(self) -> bool:
        """Check if LLM client is available and configured"""
        return self.client is not None
    
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
        """
        Get chat completion from LLM or return deterministic stub.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 - 1.0)
        
        Returns:
            Response text from LLM or stub
        """
        if not self.is_available():
            return self._stub_response(messages)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLMClient] Error during chat completion: {e}. Falling back to stub.")
            return self._stub_response(messages)
    
    def _stub_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate deterministic stub response based on message content.
        Used when LLM is unavailable.
        """
        if not messages:
            return '{"intent": "general", "confidence": 0.5, "reason": "No input provided"}'
        
        user_message = ""
        for msg in messages:
            if msg.get('role') == 'user':
                user_message = msg.get('content', '').lower()
        
        # Deterministic intent classification stub
        if 'price' in user_message or 'cost' in user_message or 'buy' in user_message:
            return '{"intent": "sales", "confidence": 0.8, "reason": "Detected pricing keywords (stub)"}'
        elif 'hello' in user_message or 'hi' in user_message or 'hey' in user_message:
            return '{"intent": "greeting", "confidence": 0.9, "reason": "Detected greeting (stub)"}'
        elif 'help' in user_message or 'support' in user_message or 'issue' in user_message:
            return '{"intent": "support", "confidence": 0.8, "reason": "Detected support keywords (stub)"}'
        else:
            return '{"intent": "general", "confidence": 0.6, "reason": "General inquiry (stub)"}'


# Memory Manager - Handles Qdrant vector database interactions
class MemoryManager:
    """
    Manages long-term memory using Qdrant vector database.
    Falls back to in-memory store if Qdrant is unavailable.
    """
    
    def __init__(self, collection_name: str = "agent_memory"):
        """
        Initialize memory manager.
        
        Args:
            collection_name: Name of the Qdrant collection
        """
        self.collection_name = collection_name
        self.client = None
        self.in_memory_store = []  # Fallback storage
        
        # Try to initialize Qdrant client
        qdrant_url = os.getenv('QDRANT_URL', 'http://localhost:6333')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Initialize client with or without API key
            if qdrant_api_key:
                self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.client = QdrantClient(url=qdrant_url)
            
            # Try to create collection if it doesn't exist
            try:
                self.client.get_collection(collection_name)
            except:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
        except ImportError:
            print("[MemoryManager] Warning: qdrant-client not installed. Using in-memory fallback.")
        except Exception as e:
            print(f"[MemoryManager] Warning: Failed to connect to Qdrant: {e}. Using in-memory fallback.")
    
    def is_available(self) -> bool:
        """Check if Qdrant client is available"""
        return self.client is not None
    
    def store(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Store text in memory with optional metadata.
        
        Args:
            text: Text content to store
            metadata: Optional metadata dict
        
        Returns:
            ID of stored item
        """
        item_id = hashlib.md5(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()
        
        if self.is_available():
            try:
                # In a real implementation, we'd generate embeddings here
                # For now, store with placeholder vector
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[{
                        "id": item_id,
                        "vector": [0.0] * 384,  # Placeholder
                        "payload": {
                            "text": text,
                            "metadata": metadata or {},
                            "timestamp": datetime.now().isoformat()
                        }
                    }]
                )
                return item_id
            except Exception as e:
                print(f"[MemoryManager] Error storing in Qdrant: {e}. Using in-memory fallback.")
        
        # Fallback to in-memory
        self.in_memory_store.append({
            "id": item_id,
            "text": text,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        })
        return item_id
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for similar items in memory.
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of matching items
        """
        if self.is_available():
            try:
                # In a real implementation, we'd use semantic search
                # For now, return empty results
                return []
            except Exception as e:
                print(f"[MemoryManager] Error searching Qdrant: {e}. Using in-memory fallback.")
        
        # Fallback: simple keyword search in in-memory store
        results = []
        query_lower = query.lower()
        for item in self.in_memory_store:
            if query_lower in item['text'].lower():
                results.append(item)
                if len(results) >= limit:
                    break
        return results


# Agent Nodes
def intent_router(user_input: str, llm_client: LLMClient, memory: MemoryManager) -> Dict[str, Any]:
    """
    Gatekeeper node: Classify user intent and route to appropriate handler.
    
    Args:
        user_input: User's message
        llm_client: LLM client instance
        memory: Memory manager instance
    
    Returns:
        Dict containing intent classification results
    """
    # Get system prompt from agent_roles or fallback
    system_prompt = get_system_prompt('gatekeeper')
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    response = llm_client.chat_completion(messages, temperature=0.3)
    
    # Parse JSON response
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        # Fallback if response is not valid JSON
        result = {
            "intent": "general",
            "confidence": 0.5,
            "reason": "Failed to parse LLM response"
        }
    
    # Store interaction in memory
    memory.store(
        text=f"User: {user_input}\nIntent: {result.get('intent', 'unknown')}",
        metadata={"type": "intent_classification", "intent": result.get('intent')}
    )
    
    return result


def librarian_node(query: str, memory: MemoryManager) -> Dict[str, Any]:
    """
    Librarian node: Retrieve relevant information from memory.
    
    Args:
        query: Search query
        memory: Memory manager instance
    
    Returns:
        Dict containing search results and summary
    """
    results = memory.search(query, limit=5)
    
    summary = f"Found {len(results)} relevant items in memory."
    if results:
        summary += "\n\nRecent interactions:\n"
        for i, item in enumerate(results[:3], 1):
            summary += f"{i}. {item['text'][:100]}...\n"
    
    return {
        "summary": summary,
        "results": results,
        "count": len(results)
    }


def sales_closer_node(user_input: str, price_data: Dict[str, Any], llm_client: LLMClient) -> str:
    """
    Closer node: Generate sales response with pricing information.
    
    Args:
        user_input: User's message
        price_data: Dictionary containing pricing information
        llm_client: LLM client instance
    
    Returns:
        Sales message string
    """
    # Get system prompt from agent_roles or fallback
    system_prompt = get_system_prompt('closer')
    
    # Build user message with pricing context
    user_message = f"""User inquiry: {user_input}

Available pricing:
{json.dumps(price_data, indent=2)}

Generate a concise sales message addressing the user's needs."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    response = llm_client.chat_completion(messages, temperature=0.7)
    return response


def support_handler(user_input: str, memory: MemoryManager) -> str:
    """
    Support handler: Process support requests.
    
    Args:
        user_input: User's support request
        memory: Memory manager instance
    
    Returns:
        Support response string
    """
    # Store support request
    memory.store(
        text=f"Support request: {user_input}",
        metadata={"type": "support", "status": "pending"}
    )
    
    # Simple acknowledgment (in real system, would route to support team)
    return ("Thank you for reaching out. I've recorded your support request. "
            "A team member will assist you shortly. Is there anything specific I can help with right now?")


def general_handler(user_input: str) -> str:
    """
    General handler: Handle general inquiries.
    
    Args:
        user_input: User's message
    
    Returns:
        General response string
    """
    return ("Thanks for your message! I'm here to help. You can ask me about:\n"
            "- Product pricing and features (sales)\n"
            "- Technical support or issues (support)\n"
            "- General information\n\n"
            "How can I assist you today?")


def greeting_handler(user_input: str) -> str:
    """
    Greeting handler: Respond to greetings.
    
    Args:
        user_input: User's greeting
    
    Returns:
        Greeting response string
    """
    return ("Hello! 👋 Welcome to our service. I'm your AI assistant, ready to help you with:\n"
            "- Product information and pricing\n"
            "- Technical support\n"
            "- Answering your questions\n\n"
            "What can I do for you today?")


def autonomous_agent_main(user_input: str, price_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Main autonomous agent workflow.
    Orchestrates the entire agent pipeline from intent classification to response generation.
    
    Args:
        user_input: User's input message
        price_data: Optional pricing data for sales queries
    
    Returns:
        Dict containing response and reasoning log
    """
    # Initialize components
    llm_client = LLMClient()
    memory = MemoryManager()
    
    # Default pricing data
    if price_data is None:
        price_data = {
            "basic": {"price": "$29/month", "features": ["Core features", "Email support"]},
            "pro": {"price": "$99/month", "features": ["All features", "Priority support", "API access"]},
            "enterprise": {"price": "Custom", "features": ["Custom solutions", "Dedicated support", "SLA"]}
        }
    
    reasoning_log = []
    
    # Step 1: Intent Classification (Gatekeeper)
    reasoning_log.append("🔍 Step 1: Classifying intent...")
    intent_result = intent_router(user_input, llm_client, memory)
    intent = intent_result.get('intent', 'general')
    confidence = intent_result.get('confidence', 0.5)
    
    reasoning_log.append(f"   Intent: {intent} (confidence: {confidence:.2f})")
    reasoning_log.append(f"   Reason: {intent_result.get('reason', 'N/A')}")
    
    # Step 2: Route to appropriate handler
    reasoning_log.append(f"\n📍 Step 2: Routing to {intent} handler...")
    
    if intent == 'sales':
        response = sales_closer_node(user_input, price_data, llm_client)
        reasoning_log.append("   ✓ Sales closer generated response")
    elif intent == 'support':
        response = support_handler(user_input, memory)
        reasoning_log.append("   ✓ Support handler processed request")
    elif intent == 'greeting':
        response = greeting_handler(user_input)
        reasoning_log.append("   ✓ Greeting handler responded")
    else:  # general
        response = general_handler(user_input)
        reasoning_log.append("   ✓ General handler provided information")
    
    # Step 3: Store interaction
    reasoning_log.append("\n💾 Step 3: Storing interaction in memory...")
    memory.store(
        text=f"User: {user_input}\nIntent: {intent}\nResponse: {response[:100]}...",
        metadata={"intent": intent, "confidence": confidence}
    )
    reasoning_log.append("   ✓ Interaction stored")
    
    return {
        "response": response,
        "intent": intent,
        "confidence": confidence,
        "reasoning_log": "\n".join(reasoning_log),
        "llm_available": llm_client.is_available(),
        "memory_available": memory.is_available()
    }


# CLI Demo
def main():
    """
    CLI demo for the autonomous agent system.
    """
    print("=" * 60)
    print("🤖 AUTONOMOUS AGENT SYSTEM - CLI Demo")
    print("=" * 60)
    print()
    
    # Check component availability
    llm_test = LLMClient()
    memory_test = MemoryManager()
    
    print("System Status:")
    print(f"  • LLM Client: {'✓ Available' if llm_test.is_available() else '✗ Unavailable (using stubs)'}")
    print(f"  • Memory Manager: {'✓ Available' if memory_test.is_available() else '✗ Unavailable (using in-memory)'}")
    print(f"  • Agent Roles: {'✓ Loaded' if AGENT_ROLES_AVAILABLE else '✗ Using fallback prompts'}")
    print()
    
    if not llm_test.is_available():
        print("⚠️  Note: GROQ_API_KEY not set. Using deterministic stubs.")
        print("   To enable LLM: export GROQ_API_KEY=your_api_key")
        print()
    
    print("Try asking:")
    print("  • 'How much does this cost?'")
    print("  • 'I need help with my account'")
    print("  • 'Hello!'")
    print("  • Type 'quit' to exit")
    print()
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Process input through agent system
            result = autonomous_agent_main(user_input)
            
            # Display response
            print(f"\n🤖 Agent: {result['response']}")
            print(f"\n📊 Metadata: Intent={result['intent']}, Confidence={result['confidence']:.2f}")
            
            # Optionally show reasoning
            show_reasoning = input("\n🔍 Show reasoning log? (y/n): ").strip().lower()
            if show_reasoning == 'y':
                print("\n" + "=" * 60)
                print("REASONING LOG:")
                print("=" * 60)
                print(result['reasoning_log'])
                print("=" * 60)
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
