"""
Advanced persona system for compensation planning.
Provides structured persona configurations for different user roles.
"""
from typing import Dict, List, Any, Optional

# Enhanced persona system with detailed attributes
PERSONAS = {
    "Recruitment Manager": {
        "style": "analytical and data-driven",
        "risk_tolerance": "moderate",
        "perspective": "market-focused",
        "confidence_threshold": 0.7,
        "prompt_prefix": "You are a compensation expert focused on market data and trends. Provide detailed analysis with justification. You create comprehensive compensation packages based on role, level, and location.",
        "tone": "objective",
        "detail_level": "high",
        "data_emphasis": True,
        "risk_emphasis": True,
        "policy_emphasis": True
    },
    "HR Director": {
        "style": "policy-oriented and holistic",
        "risk_tolerance": "moderate",
        "perspective": "employee-centered",
        "confidence_threshold": 0.65,
        "prompt_prefix": "You are an HR Director balancing policy compliance with competitive compensation. Focus on equity, compliance and retention.",
        "tone": "balanced",
        "detail_level": "medium",
        "data_emphasis": True,
        "risk_emphasis": True,
        "policy_emphasis": True
    },
    "Hiring Manager": {
        "style": "practical and goal-oriented",
        "risk_tolerance": "higher",
        "perspective": "talent-acquisition focused",
        "confidence_threshold": 0.6,
        "prompt_prefix": "You are a hiring manager trying to attract top talent while managing your budget. Balance candidate appeal with business constraints.",
        "tone": "confident",
        "detail_level": "medium",
        "data_emphasis": False,
        "risk_emphasis": False,
        "policy_emphasis": True
    }
}

def get_persona_config(persona_name: str = "Recruitment Manager") -> Dict[str, Any]:
    """Get the configuration for a specific persona."""
    if persona_name in PERSONAS:
        return PERSONAS[persona_name]
    
    # Return default Recruitment Manager persona if requested one not found
    return PERSONAS["Recruitment Manager"]

def get_persona_names() -> List[str]:
    """Get list of available persona names."""
    return list(PERSONAS.keys())

def get_persona_prompt(persona_name: str, query: str) -> str:
    """Generate a persona-specific prompt for the query."""
    persona = get_persona_config(persona_name)
    
    # Craft a prompt that incorporates the persona's characteristics
    prompt = f"""{persona['prompt_prefix']}

When answering this compensation question, remember to:
- Use a {persona['tone']} tone
- Provide {'detailed' if persona['detail_level'] == 'high' else 'concise'} analysis
{f"- Emphasize market data and benchmarks" if persona['data_emphasis'] else ""}
{f"- Highlight potential risks and challenges" if persona['risk_emphasis'] else ""}
{f"- Consider policy and compliance implications" if persona['policy_emphasis'] else ""}
- Reflect a {persona['risk_tolerance']} risk tolerance approach
- Focus on {persona['perspective']} aspects

User Query: {query}

Response:"""
    
    return prompt

def apply_persona_to_tools(persona_name: str, tools_plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adjust tool execution plan based on persona preferences."""
    persona = get_persona_config(persona_name)
    modified_plan = []
    
    # Apply persona-specific modifications to each tool in the plan
    for tool in tools_plan:
        tool_name = tool.get("tool", "")
        
        # Make a copy of the tool to modify
        modified_tool = tool.copy()
        
        # Enhance args with persona context
        if "args" in modified_tool:
            modified_tool["args"]["persona"] = persona_name
            modified_tool["args"]["risk_tolerance"] = persona["risk_tolerance"]
            
            # For risk analysis tools, adjust thresholds based on persona
            if tool_name == "flag_risk" or tool_name == "risk_analyzer":
                if persona["risk_tolerance"] == "low":
                    modified_tool["args"]["risk_threshold"] = 0.3  # Lower threshold = more risks flagged
                elif persona["risk_tolerance"] == "high":
                    modified_tool["args"]["risk_threshold"] = 0.7  # Higher threshold = fewer risks flagged
                else:
                    modified_tool["args"]["risk_threshold"] = 0.5  # Moderate default
            
            # For policy tools, adjust strictness based on persona
            if tool_name == "check_policy":
                if persona["policy_emphasis"]:
                    modified_tool["args"]["strictness"] = "high"
                else:
                    modified_tool["args"]["strictness"] = "moderate"
        
        modified_plan.append(modified_tool)
    
    # Add persona-specific tools based on the persona
    if persona_name == "HR Director":
        # HR Director gets policy compliance analysis
        modified_plan.append({
            "tool": "policy_compliance",
            "args": {
                "persona": persona_name,
                "analysis_depth": "detailed"
            }
        })
    
    if persona_name == "Hiring Manager":
        # Hiring Manager gets budget impact analysis
        modified_plan.append({
            "tool": "budget_impact",
            "args": {
                "persona": persona_name,
                "focus": "talent-retention"
            }
        })
    
    return modified_plan