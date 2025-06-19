"""
Advanced Memory Manager for Agentic Compensation Planning.
Handles conversation context, user preferences, and learning from feedback.
"""
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from cachetools import TTLCache
from pydantic import BaseModel
import tiktoken

class ConversationMemory(BaseModel):
    """Structured conversation memory with context carryover."""
    session_id: str
    user_preferences: Dict[str, Any] = {}
    conversation_history: List[Dict[str, Any]] = []
    learned_patterns: Dict[str, Any] = {}
    last_interaction: datetime
    context_summary: str = ""
    
class UserPreferences(BaseModel):
    """User-specific preferences learned over time."""
    preferred_locations: List[str] = []
    typical_roles: List[str] = []
    company_stages: List[str] = []
    budget_ranges: Dict[str, int] = {}
    communication_style: str = "detailed"  # brief, detailed, technical
    risk_tolerance: str = "medium"  # low, medium, high

class MemoryManager:
    """Advanced memory management for agentic AI system."""
    
    def __init__(self, cache_ttl: int = 3600):  # 1 hour TTL
        self.conversation_cache = TTLCache(maxsize=100, ttl=cache_ttl)
        self.user_preferences_cache = TTLCache(maxsize=50, ttl=cache_ttl * 24)  # 24 hours
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def get_conversation_memory(self, session_id: str) -> ConversationMemory:
        """Retrieve or create conversation memory for a session."""
        if session_id not in self.conversation_cache:
            self.conversation_cache[session_id] = ConversationMemory(
                session_id=session_id,
                last_interaction=datetime.now()
            )
        return self.conversation_cache[session_id]
    
    def update_conversation(self, session_id: str, query: str, response: Dict[str, Any]):
        """Update conversation history with new interaction."""
        memory = self.get_conversation_memory(session_id)
        
        # Add interaction to history
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "query_type": response.get("query_type", "unknown")
        }
        memory.conversation_history.append(interaction)
        
        # Learn from the interaction
        self._learn_from_interaction(memory, interaction)
        
        # Summarize if history gets too long
        if len(memory.conversation_history) > 10:
            memory.context_summary = self._summarize_context(memory)
            # Keep only recent interactions
            memory.conversation_history = memory.conversation_history[-5:]
        
        memory.last_interaction = datetime.now()
        self.conversation_cache[session_id] = memory
    
    def _learn_from_interaction(self, memory: ConversationMemory, interaction: Dict[str, Any]):
        """Learn patterns from user interactions."""
        query = interaction["query"].lower()
        response = interaction["response"]
        
        # Learn location preferences
        locations = ["san francisco", "new york", "seattle", "toronto", "austin", "boston"]
        for loc in locations:
            if loc in query and loc not in memory.user_preferences.get("preferred_locations", []):
                if "preferred_locations" not in memory.user_preferences:
                    memory.user_preferences["preferred_locations"] = []
                memory.user_preferences["preferred_locations"].append(loc)
        
        # Learn role patterns
        roles = ["software engineer", "product manager", "designer", "data scientist", "director"]
        for role in roles:
            if role in query and role not in memory.user_preferences.get("typical_roles", []):
                if "typical_roles" not in memory.user_preferences:
                    memory.user_preferences["typical_roles"] = []
                memory.user_preferences["typical_roles"].append(role)
        
        # Learn communication preferences
        if "brief" in query or "quick" in query or "summarize" in query:
            memory.user_preferences["communication_style"] = "brief"
        elif "detailed" in query or "comprehensive" in query or "thorough" in query:
            memory.user_preferences["communication_style"] = "detailed"
    
    def _summarize_context(self, memory: ConversationMemory) -> str:
        """Create a concise summary of conversation context."""
        if not memory.conversation_history:
            return ""
        
        # Extract key themes and patterns
        recent_queries = [item["query"] for item in memory.conversation_history[-5:]]
        query_types = [item.get("query_type", "unknown") for item in memory.conversation_history[-5:]]
        
        summary = f"Recent conversation focused on {', '.join(set(query_types))}. "
        summary += f"User preferences: {memory.user_preferences}. "
        summary += f"Recent topics: {', '.join(recent_queries[:3])}."
        
        return summary
    
    def get_relevant_context(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Get relevant context for current query."""
        memory = self.get_conversation_memory(session_id)
        
        context = {
            "user_preferences": memory.user_preferences,
            "conversation_summary": memory.context_summary,
            "recent_interactions": memory.conversation_history[-3:],
            "learned_patterns": memory.learned_patterns
        }
        
        return context
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text for context management."""
        return len(self.encoding.encode(text))
    
    def optimize_context_for_tokens(self, context: Dict[str, Any], max_tokens: int = 2000) -> Dict[str, Any]:
        """Optimize context to fit within token limits."""
        context_str = json.dumps(context)
        current_tokens = self.count_tokens(context_str)
        
        if current_tokens <= max_tokens:
            return context
        
        # Progressively reduce context
        optimized = context.copy()
        
        # First, reduce recent interactions
        if "recent_interactions" in optimized and len(optimized["recent_interactions"]) > 1:
            optimized["recent_interactions"] = optimized["recent_interactions"][-1:]
        
        # Then, truncate summary if still too long
        if self.count_tokens(json.dumps(optimized)) > max_tokens:
            if "conversation_summary" in optimized:
                summary = optimized["conversation_summary"]
                if len(summary) > 200:
                    optimized["conversation_summary"] = summary[:200] + "..."
        
        return optimized
    
    def clear_session(self, session_id: str):
        """Clear memory for a specific session."""
        if session_id in self.conversation_cache:
            del self.conversation_cache[session_id]
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        memory = self.get_conversation_memory(session_id)
        
        return {
            "total_interactions": len(memory.conversation_history),
            "session_duration": (datetime.now() - memory.last_interaction).total_seconds() / 60,
            "learned_preferences": len(memory.user_preferences),
            "query_types": list(set([item.get("query_type", "unknown") for item in memory.conversation_history]))
        }