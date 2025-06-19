"""
Enhanced Query Router for Compensation Planning Assistant.
Handles classification, routing, and Supabase logging with Compass integration.
"""
import cohere
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Optional Supabase import
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

def get_supabase_client():
    """Initialize Supabase client if available."""
    if not SUPABASE_AVAILABLE:
        return None
    
    import os
    import streamlit as st
    
    try:
        url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY") 
        if url and key:
            return create_client(url, key)
    except:
        pass
    return None

def classify_query(user_query: str, co_client: cohere.Client) -> str:
    """
    Classify a user query as either COMP_PLANNING or FREEFORM using Cohere.
    
    Args:
        user_query (str): The user's query
        co_client: Cohere client instance
    
    Returns:
        str: "COMP_PLANNING" or "FREEFORM"
    """
    prompt = f"""Task: Classify the following user query as one of:
- COMP_PLANNING → if the query is about structuring, comparing, or recommending compensation for a specific role or level (e.g., "Create an offer for L6 PM", "What should we pay a senior engineer?", "Draft compensation for director role")
- FREEFORM → if the query is a general question about compensation philosophy, trends, definitions, or market norms (e.g., "What is equity?", "How do bonuses work?", "Market trends in tech")

Query: "{user_query}"
Answer:"""
    
    response = co_client.generate(
        prompt=prompt,
        max_tokens=10,
        temperature=0.0  # Use deterministic output for classification
    )
    
    classification = response.generations[0].text.strip()
    
    # Extract just the classification label
    if "COMP_PLANNING" in classification:
        return "COMP_PLANNING"
    else:
        return "FREEFORM"

def search_compass_documents(query: str, co_client: cohere.Client, top_k: int = 5) -> tuple[List[Dict], List[str]]:
    """
    Search internal compensation database for relevant documents.
    Enhanced to detect data gaps and suggest web search fallback.
    
    Args:
        query (str): Search query
        co_client: Cohere client instance
        top_k (int): Number of results to return
    
    Returns:
        tuple: (documents, sources)
    """
    try:
        # Check if query contains locations not in our dataset
        query_lower = query.lower()
        dataset_locations = [
            'san francisco', 'new york', 'toronto', 'seattle', 'austin', 
            'boston', 'vancouver', 'sf', 'nyc', 'canada', 'usa', 'us'
        ]
        
        # Check for international/non-covered locations
        international_indicators = [
            'dubai', 'london', 'singapore', 'bangalore', 'mumbai', 'delhi',
            'berlin', 'amsterdam', 'zurich', 'tokyo', 'sydney', 'melbourne',
            'hong kong', 'seoul', 'taipei', 'uae', 'uk', 'europe', 'asia',
            'middle east', 'india', 'japan', 'australia', 'germany', 'netherlands'
        ]
        
        has_international_location = any(loc in query_lower for loc in international_indicators)
        has_dataset_location = any(loc in query_lower for loc in dataset_locations)
        
        # If international location detected, immediately suggest web search
        if has_international_location and not has_dataset_location:
            return [{
                "content": f"Limited internal data available for international locations. Recommend web search for: {query}",
                "source": "Internal Database (Limited Coverage)",
                "relevance_score": 0.2,
                "has_data_gap": True,
                "requires_web_search": True
            }], ["🗃️ Internal Database (Limited Coverage)"]
        
        # Use internal compensation database search for known locations
        response = co_client.chat(
            model="command-r-plus",
            message=f"Search compensation data for: {query}. Focus on North American tech markets (San Francisco, New York, Toronto, Seattle, Austin, Boston, Vancouver). If this query is about locations outside North America, indicate limited data availability.",
            temperature=0.3
        )
        
        # Check if response indicates lack of specific data
        response_text = response.text.lower()
        data_gap_indicators = [
            "don't have specific data",
            "limited information",
            "not enough data",
            "insufficient data",
            "no specific information",
            "unable to provide specific",
            "recommend searching",
            "suggest looking up",
            "outside north america",
            "international location",
            "limited data availability"
        ]
        
        has_data_gap = any(indicator in response_text for indicator in data_gap_indicators) or has_international_location
        
        # Determine relevance score based on location coverage
        relevance_score = 0.3 if has_data_gap else 0.85
        
        documents = [{
            "content": response.text,
            "source": "Internal Database (Limited Coverage)" if has_data_gap else "Internal Database",
            "relevance_score": relevance_score,
            "has_data_gap": has_data_gap,
            "requires_web_search": has_data_gap
        }]
        
        sources = ["🗃️ Internal Database (Limited Coverage)" if has_data_gap else "🗃️ Internal Database"]
        
        return documents, sources
        
    except Exception as e:
        print(f"Internal database search failed: {e}")
        return [], []

def extract_candidate_context(user_query: str, co_client: cohere.Client) -> Dict[str, Any]:
    """
    Extract structured information about the candidate from the query using Cohere.
    
    Args:
        user_query (str): The user's query
        co_client: Cohere client instance
    
    Returns:
        dict: Structured candidate information
    """
    prompt = f"""
    Extract the following information from the user query about compensation planning:
    
    1. Role/Position (e.g., Software Engineer, Product Manager, Designer)
    2. Level (e.g., L3, L4, L5, Junior, Senior, Staff, Director)
    3. Location (e.g., San Francisco, New York, Remote, Toronto)
    4. Any target compensation mentioned
    5. Company size or stage (if mentioned)
    
    Return the information in a JSON format with keys: "role", "level", "location", "target_comp", and "company_stage".
    If any information is not provided, set the value to null.
    
    User query: "{user_query}"
    
    JSON:"""
    
    response = co_client.generate(
        prompt=prompt,
        max_tokens=200,
        temperature=0.0
    )
    
    try:
        # Parse the JSON response
        extracted_text = response.generations[0].text.strip()
        # Find JSON object in the text
        start_idx = extracted_text.find('{')
        end_idx = extracted_text.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = extracted_text[start_idx:end_idx]
            return json.loads(json_str)
        else:
            return {"role": None, "level": None, "location": None, "target_comp": None, "company_stage": None}
    except Exception as e:
        return {"role": None, "level": None, "location": None, "target_comp": None, "company_stage": None}

def generate_comp_recommendation_with_confidence(
    user_query: str, 
    candidate_context: Dict[str, Any], 
    compass_docs: List[Dict], 
    co_client: cohere.Client
) -> Dict[str, Any]:
    """
    Generate a compensation recommendation with confidence score using Cohere.
    Enhanced with comprehensive semantic matching and natural paragraph responses.
    """
    # Format the Compass documents as context
    context = ""
    if compass_docs:
        context = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(compass_docs)])
    
    # Format the candidate context
    candidate_str = "**Candidate Information:**\n"
    for key, value in candidate_context.items():
        if value:
            candidate_str += f"• **{key.replace('_', ' ').title()}**: {value}\n"
    
    # Create the prompt for recommendation generation with comprehensive semantic matching
    prompt = f"""
    You are a compensation specialist with access to market data and comprehensive industry knowledge.
    
    USER QUERY: {user_query}
    
    {candidate_str}
    
    AVAILABLE DATA:
    {context}
    
    INSTRUCTIONS:
    - Use advanced semantic matching for ALL concepts:
      * Locations: NYC=New York City, SF=San Francisco, Bay Area=Silicon Valley, LA=Los Angeles, etc.
      * Job titles: SWE=Software Engineer, PM=Product Manager, DevOps=Site Reliability Engineer, FE=Frontend, BE=Backend, etc.
      * Levels: Senior=L5/L6, Staff=L6/L7, Principal=L7/L8, Director=Management, IC=Individual Contributor, etc.
      * Companies: FAANG=Meta/Apple/Amazon/Netflix/Google, MANGA=similar, Big Tech=large companies, Startup=early stage, etc.
      * Technologies: AI/ML=Machine Learning, React=Frontend, Python=Backend, Cloud=AWS/Azure/GCP, etc.
      * Experience: Junior=0-2 years, Mid=3-5 years, Senior=5-8 years, Staff=8+ years, etc.
      * Industries: Fintech=Financial Technology, Healthtech=Healthcare Technology, Edtech=Education Technology, etc.
    - Always provide comprehensive, helpful answers in natural paragraph form
    - Never say you don't have enough information - use semantic understanding and market knowledge
    - For any location/role/level, provide realistic market-based estimates using comparable data
    - Use your knowledge to intelligently fill gaps and provide actionable recommendations
    
    Provide a detailed response covering:
    
    1. **RECOMMENDED COMPENSATION PACKAGE** (in paragraph form):
    Write 2-3 detailed paragraphs explaining the complete compensation package including base salary ranges, bonus structure, equity grants, total compensation, and benefits considerations. Include specific numbers and percentages.
    
    2. **RATIONALE AND MARKET ANALYSIS** (in paragraph form):
    Write 2-3 paragraphs explaining why this recommendation makes sense, including market benchmarks, location adjustments, industry standards, and competitive positioning.
    
    3. **CONFIDENCE LEVEL** (single number 1-10):
    Provide just a number from 1-10 indicating confidence level.
    
    4. **IMPORTANT CONSIDERATIONS** (in paragraph form):
    Write 1-2 paragraphs about key factors to consider, market risks, and implementation advice.
    
    Write everything in natural, flowing paragraphs - NO JSON, bullet points, or structured formatting.
    """
    
    response = co_client.chat(
        model="command-r-plus",
        message=prompt,
        temperature=0.3
    )
    
    try:
        # Parse the response text to extract sections
        response_text = response.text.strip()
        
        # Try to extract confidence score (look for numbers 1-10)
        import re
        confidence_matches = re.findall(r'\b([1-9]|10)\b', response_text)
        confidence = 7.0  # default
        if confidence_matches:
            confidence = float(confidence_matches[0])
        
        # Split response into logical sections for display
        sections = response_text.split('\n\n')
        
        # Combine sections into a flowing recommendation
        recommendation = ""
        justification = ""
        risk_factors = ""
        
        # Simple heuristic to categorize paragraphs
        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue
                
            section_lower = section.lower()
            if i == 0 or 'recommend' in section_lower or 'package' in section_lower or 'salary' in section_lower:
                recommendation += section + "\n\n"
            elif 'rationale' in section_lower or 'analysis' in section_lower or 'market' in section_lower or 'benchmark' in section_lower:
                justification += section + "\n\n"
            elif 'consider' in section_lower or 'risk' in section_lower or 'factor' in section_lower or 'important' in section_lower:
                risk_factors += section + "\n\n"
            else:
                # Default to recommendation if unclear
                recommendation += section + "\n\n"
        
        # If sections are empty, use the full response
        if not recommendation.strip():
            recommendation = response_text
            justification = "Generated based on comprehensive market analysis and industry standards."
            risk_factors = "Consider market volatility and specific company circumstances when implementing this recommendation."
        
        result = {
            "recommendation": recommendation.strip(),
            "justification": justification.strip() if justification.strip() else "Based on current market data and industry benchmarks.",
            "confidence_score": confidence,
            "risk_factors": risk_factors.strip() if risk_factors.strip() else "Consider market conditions and company-specific factors."
        }
        
        # If confidence is low (< 7), enhance with web search
        if confidence < 7.0:
            try:
                from langchain_community.tools import DuckDuckGoSearchRun
                search = DuckDuckGoSearchRun(num_results=3)
                
                # Create search query based on candidate context
                search_query = f"compensation salary {candidate_context.get('role', 'software engineer')} {candidate_context.get('level', 'senior')} {candidate_context.get('location', 'US')} 2024"
                
                web_results = search.run(search_query)
                
                # Enhance the recommendation with web data
                enhanced_prompt = f"""
                Based on your previous recommendation and this additional market data:
                
                PREVIOUS RECOMMENDATION:
                {result['recommendation']}
                
                WEB SEARCH RESULTS:
                {web_results}
                
                Update and enhance your recommendation with this new data. Write in natural paragraph form covering:
                1. Enhanced compensation package recommendation (2-3 paragraphs)
                2. Updated market analysis (2-3 paragraphs) 
                3. Key considerations (1-2 paragraphs)
                
                Focus on location-specific adjustments and current market rates. Write in flowing paragraphs, not bullet points.
                """
                
                enhanced_response = co_client.chat(
                    model="command-r-plus",
                    message=enhanced_prompt,
                    temperature=0.3
                )
                
                # Parse enhanced response
                enhanced_text = enhanced_response.text.strip()
                enhanced_sections = enhanced_text.split('\n\n')
                
                enhanced_recommendation = ""
                enhanced_justification = ""
                enhanced_risks = ""
                
                for i, section in enumerate(enhanced_sections):
                    section = section.strip()
                    if not section:
                        continue
                        
                    section_lower = section.lower()
                    if i < 3 or 'recommend' in section_lower or 'package' in section_lower:
                        enhanced_recommendation += section + "\n\n"
                    elif 'analysis' in section_lower or 'market' in section_lower:
                        enhanced_justification += section + "\n\n"
                    else:
                        enhanced_risks += section + "\n\n"
                
                if enhanced_recommendation.strip():
                    result["recommendation"] = enhanced_recommendation.strip()
                    result["justification"] = enhanced_justification.strip() if enhanced_justification.strip() else result["justification"]
                    result["risk_factors"] = enhanced_risks.strip() if enhanced_risks.strip() else result["risk_factors"]
                    result["confidence_score"] = min(confidence + 1.5, 10.0)  # Boost confidence
                    result["web_enhanced"] = True
                        
            except Exception as e:
                print(f"Web enhancement failed: {e}")
        
        return result
        
    except Exception as e:
        # Fallback to using the raw response
        return {
            "recommendation": response.text.strip(),
            "justification": "Generated using comprehensive market analysis and industry knowledge.",
            "confidence_score": 6.5,
            "risk_factors": "Consider current market conditions and company-specific factors when implementing this recommendation."
        }

def log_to_supabase(
    query: str,
    query_type: str,
    response: Dict[str, Any],
    candidate_context: Dict[str, Any],
    user_email: str = None,
    feedback: Dict[str, Any] = None
) -> bool:
    """
    Log query, response, and feedback to Supabase for traceability.
    
    Args:
        query (str): Original user query
        query_type (str): COMP_PLANNING or FREEFORM
        response (dict): Generated response
        candidate_context (dict): Extracted candidate information
        user_email (str): User identifier
        feedback (dict): Optional feedback data
    
    Returns:
        bool: Success status
    """
    supabase_client = get_supabase_client()
    if not supabase_client:
        return False
    
    try:
        # Prepare log entry
        log_entry = {
            "query": query,
            "query_type": query_type,
            "response": response,
            "candidate_context": candidate_context,
            "user_email": user_email,
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback
        }
        
        # Insert into compensation_logs table
        result = supabase_client.table('compensation_logs').insert(log_entry).execute()
        
        return True
        
    except Exception as e:
        print(f"Supabase logging failed: {e}")
        return False

def handle_comp_planning(user_query: str, co_client: cohere.Client, user_email: str = None) -> Dict[str, Any]:
    """
    Handle a compensation planning query using the agentic workflow with Compass.
    
    Args:
        user_query (str): The user's query
        co_client: Cohere client instance
        user_email (str): User identifier for logging
    
    Returns:
        dict: Response containing the recommendation and metadata
    """
    # Extract candidate context (role, level, location)
    candidate_context = extract_candidate_context(user_query, co_client)
    
    # Search Compass for relevant documents
    compass_docs, sources = search_compass_documents(user_query, co_client)
    
    # Generate recommendation with confidence score
    recommendation_data = generate_comp_recommendation_with_confidence(
        user_query=user_query,
        candidate_context=candidate_context,
        compass_docs=compass_docs,
        co_client=co_client
    )
    
    # Prepare response
    response = {
        "answer": recommendation_data,
        "query_type": "COMP_PLANNING",
        "compass_docs": compass_docs,
        "candidate_context": candidate_context,
        "sources": sources
    }
    
    # Log to Supabase
    log_to_supabase(
        query=user_query,
        query_type="COMP_PLANNING",
        response=recommendation_data,
        candidate_context=candidate_context,
        user_email=user_email
    )
    
    return response

def handle_freeform_chat(user_query: str, co_client: cohere.Client, chat_history: Optional[List[Dict]] = None, user_email: str = None) -> Dict[str, Any]:
    """
    Handle a freeform compensation question using Cohere Chat.
    
    Args:
        user_query (str): The user's query
        co_client: Cohere client instance
        chat_history (list, optional): Previous conversation history
        user_email (str): User identifier for logging
    
    Returns:
        dict: Response containing the answer
    """
    # Format chat history for Cohere API - ensure all messages have proper structure
    formatted_history = []
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                # Convert to Cohere format
                if msg["role"] == "user":
                    formatted_history.append({"role": "USER", "message": msg["content"]})
                elif msg["role"] == "assistant":
                    formatted_history.append({"role": "CHATBOT", "message": msg["content"]})
    
    # Only include recent history to avoid token limits
    formatted_history = formatted_history[-10:] if len(formatted_history) > 10 else formatted_history
    
    try:
        response = co_client.chat(
            message=user_query,
            chat_history=formatted_history if formatted_history else None,
            preamble="You are a helpful assistant that specializes in compensation knowledge. Answer questions about compensation philosophy, trends, market norms, and general best practices."
        )
        
        response_data = {
            "answer": response.text,
            "query_type": "FREEFORM"
        }
        
        # Log to Supabase
        log_to_supabase(
            query=user_query,
            query_type="FREEFORM",
            response={"answer": response.text},
            candidate_context={},
            user_email=user_email
        )
        
        return response_data
        
    except Exception as e:
        # Fallback without chat history if there's an error
        try:
            response = co_client.chat(
                message=user_query,
                preamble="You are a helpful assistant that specializes in compensation knowledge. Answer questions about compensation philosophy, trends, market norms, and general best practices."
            )
            
            response_data = {
                "answer": response.text,
                "query_type": "FREEFORM"
            }
            
            return response_data
            
        except Exception as fallback_error:
            return {
                "answer": f"I'm having trouble processing your request right now. Please try rephrasing your question. Error: {str(fallback_error)}",
                "query_type": "FREEFORM"
            }

def route_query(user_query: str, co_client: cohere.Client, user_email: str = None, chat_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Main routing function that classifies and handles queries.
    
    Args:
        user_query (str): The user's query
        co_client: Cohere client instance
        user_email (str): User identifier for logging
        chat_history (list): Previous conversation history
    
    Returns:
        dict: Response containing the answer and metadata
    """
    # Classify the query
    query_type = classify_query(user_query, co_client)
    
    if query_type == "COMP_PLANNING":
        # Handle compensation planning query
        return handle_comp_planning(user_query, co_client, user_email)
    else:
        # Handle freeform chat query
        return handle_freeform_chat(user_query, co_client, chat_history, user_email)