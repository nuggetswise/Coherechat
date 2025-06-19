"""
Compatibility patch for Cohere SDK 5.15.0+ and langchain-cohere.
This provides the ToolResult class that langchain-cohere expects.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Check if ToolResult exists in cohere.types
try:
    from cohere.types import ToolResult
except ImportError:
    # If not, create our own compatible version
    from cohere.types import ToolCall
    
    class ToolResult(BaseModel):
        """Compatible ToolResult class for newer Cohere SDK versions."""
        call: ToolCall
        outputs: List[Dict[str, Any]] = Field(default_factory=list)
        
        def dict(self) -> Dict[str, Any]:
            """Convert to dictionary for API compatibility."""
            return {
                "call": self.call.dict() if hasattr(self.call, "dict") else self.call,
                "outputs": self.outputs
            }
