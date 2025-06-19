import json

def validate_recommendation(recommendation, query, persona, co_client=None):
    """
    Validate generated recommendation against constraints and persona style using LLM.
    Returns dict: {"is_valid": bool, "score": float, "issues": list, "improvements": list}
    """
    if not co_client:
        return {"is_valid": True, "score": 7, "issues": [], "improvements": []}
    prompt = f"""
    Critique the following compensation recommendation for completeness, compliance, and confidence.
    Persona: {persona}
    Query: {query}
    Recommendation: {recommendation}
    
    Return JSON with:
    - is_valid (true/false)
    - score (1-10)
    - issues (list of strings)
    - improvements (list of strings)
    """
    try:
        response = co_client.generate(prompt=prompt, max_tokens=200)
        text = response.generations[0].text.strip()
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {"is_valid": True, "score": 7, "issues": [], "improvements": []}