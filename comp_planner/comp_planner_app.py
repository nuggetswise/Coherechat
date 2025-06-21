"""
Enhanced Compensation Planner Application with AI-Powered Analysis.
Integrates RAG system, evaluation framework, and enhanced UI.
"""
import streamlit as st
import cohere
import os
import sys
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

# Import our new enhanced components
try:
    from comp_planner.cohere_rag_system import get_cohere_rag_system
except ImportError:
    def get_cohere_rag_system(client):
        st.warning("RAG system not available")
        return None

from comp_planner.evaluation_framework import get_compensation_evaluator, get_agent_consensus_analyzer
from comp_planner.enhanced_ui import (
    display_enhanced_sidebar, create_evaluation_dashboard_ui, 
    display_recommendation_comparison, create_export_functionality,
    display_agent_workflow_progress, display_example_prompts
)

# Import existing modules
from comp_planner import auth
from comp_planner import query_router
from agents.offer_chain import run_compensation_planner
from comp_planner.tool_manager import run_tool_action

def get_api_keys():
    """Get API keys from environment or structured secrets."""
    cohere_api_key = os.environ.get("COHERE_API_KEY", "")
    
    if not cohere_api_key and hasattr(st, "secrets") and "cohere" in st.secrets:
        cohere_api_key = st.secrets.cohere.COHERE_API_KEY
    
    return cohere_api_key

def initialize_cohere_client():
    """Initialize Cohere client with API key."""
    api_key = get_api_keys()
    if api_key:
        return cohere.Client(api_key)
    else:
        st.error("🔑 Cohere API key not found in secrets or environment variables")
        st.info("Please configure COHERE_API_KEY in your .streamlit/secrets.toml file")
        st.stop()
    return None

def initialize_enhanced_systems(cohere_client):
    """Initialize all enhanced systems (CrewAI, RAG, Evaluation)"""
    systems = {}
    
    try:
        # Initialize RAG system
        systems["rag"] = get_cohere_rag_system(cohere_client)
        st.session_state["rag_system"] = systems["rag"]
        
        # Initialize evaluation systems
        systems["evaluator"] = get_compensation_evaluator()
        systems["consensus_analyzer"] = get_agent_consensus_analyzer()
        st.session_state["evaluator"] = systems["evaluator"]
        st.session_state["consensus_analyzer"] = systems["consensus_analyzer"]
        
        return systems
        
    except Exception as e:
        st.error(f"Error initializing enhanced systems: {e}")
        return {}

def execute_rag_enhanced_query(query: str, role: str, level: str, location: str, rag_system) -> Dict[str, Any]:
    """Execute RAG-enhanced compensation query"""
    
    with st.spinner("🔍 Searching compensation database with RAG..."):
        
        # Show RAG process
        with st.expander("🔍 RAG Process Details", expanded=True):
            st.markdown("**Step 1**: Semantic search for relevant compensation data...")
            
            # Execute RAG query
            result = rag_system.generate_compensation_recommendation(
                query=query,
                role=role, 
                level=level,
                location=location
            )
            
            st.markdown("**Step 2**: Reranking results using Cohere Rerank...")
            st.markdown("**Step 3**: Generating recommendation with retrieved context...")
            
            # Show sources used
            if result.get("sources"):
                st.markdown("**Sources used:**")
                for source in result["sources"]:
                    st.markdown(f"• {source}")
            
            st.success(f"✅ Generated recommendation using {result.get('benchmark_count', 0)} benchmarks")
        
        return result

def main():
    """Enhanced main function with integrated multi-agent workflow and automatic evaluation"""
    
    # Initialize Cohere client
    if 'cohere_client' not in st.session_state:
        st.error("Cohere client not initialized. Please go back and configure API key.")
        return
    
    co_client = st.session_state.cohere_client
    
    # Initialize enhanced systems
    if 'systems_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing compensation planning systems..."):
            systems = initialize_enhanced_systems(co_client)
            st.session_state['systems_initialized'] = True
            st.success("✅ All systems initialized successfully!")
    
    # Enhanced sidebar with advanced controls
    sidebar_config = display_enhanced_sidebar()
    
    # App header - focused on Multi-Agent with evaluation
    st.title("💰 Enhanced Compensation Planner")
    st.caption("🤖 Multi-Agent System with AI-Powered Evaluation")
    
    # Display the integrated Multi-Agent interface
    display_integrated_interface(co_client, sidebar_config)

def display_integrated_interface(co_client, config):
    """Display integrated interface with Multi-Agent workflow and automatic evaluation"""
    
    st.markdown("### 🤖 Multi-Agent Compensation Planning")
    st.info("This system uses three specialized agents to create comprehensive compensation recommendations with automatic AI evaluation")
    
    # Display example prompts
    display_example_prompts()
    
    # Input form for compensation request
    with st.form("multi_agent_request"):
        st.markdown("**Compensation Request Details:**")
        # Move text input to the top
        query_text = st.text_area(
            "Describe your compensation request (required)",
            value=st.session_state.get("multi_agent_query_text", ""),
            placeholder="e.g. Create a compensation package for a Junior Software Engineer in San Francisco",
            key="multi_agent_query_text"
        )
        st.markdown("---")
        st.markdown("**Optional: Fill in details below or leave blank**")
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("Role/Position", value="", help="e.g., Software Engineer, Product Manager", key="multi_agent_role")
            level = st.selectbox("Seniority Level", ["", "Junior", "Mid", "Senior", "Staff", "Principal", "Director"], key="multi_agent_level")
        with col2:
            location = st.text_input("Location", value="", help="City or Remote", key="multi_agent_location")
            department = st.text_input("Department", value="", help="Department or team", key="multi_agent_department")
        submitted = st.form_submit_button("🚀 Generate Compensation Plan", use_container_width=True)
    
    if submitted and query_text:
        # Create context from form inputs or extract from query
        context = {
            "role": role or "Software Engineer",
            "level": level or "Junior", 
            "location": location or "San Francisco",
            "department": department or "Engineering",
            "query": query_text
        }
        
        # Execute multi-agent workflow
        with st.spinner("🔄 Running multi-agent compensation analysis..."):
            workflow_result = execute_multi_agent_workflow_fallback(query_text, context, co_client)
        
        # Display workflow progress
        display_agent_workflow_progress(workflow_result)
        
        if workflow_result.get("success", False):
            st.success("🎉 Multi-Agent workflow completed successfully!")
            
            # Display the final recommendation
            st.markdown("### 📋 Final Compensation Recommendation")
            workflow_output = workflow_result.get("workflow_result", "")
            if workflow_output:
                st.markdown(workflow_output)
            
            # Display individual agent outputs
            if "agent_outputs" in workflow_result:
                with st.expander("🔍 View Individual Agent Analysis", expanded=False):
                    agent_outputs = workflow_result["agent_outputs"]
                    
                    tab1, tab2, tab3 = st.tabs(["👤 Recruitment Manager", "🏢 HR Director", "👔 Hiring Manager"])
                    
                    with tab1:
                        st.markdown("**Draft Compensation Package:**")
                        st.markdown(agent_outputs.get("recruitment_manager", "No output available"))
                    
                    with tab2:
                        st.markdown("**Policy Validation & Review:**")
                        st.markdown(agent_outputs.get("hr_director", "No output available"))
                    
                    with tab3:
                        st.markdown("**Final Approval Decision:**")
                        st.markdown(agent_outputs.get("hiring_manager", "No output available"))
            
            # Auto-evaluation - always enabled
            evaluator = st.session_state.get("evaluator")
            if evaluator:
                # Pass enhanced context for new AI dimensions
                enhanced_context = {
                    **context,
                    "key_points": [f"{level} {role}", location, department],
                    "supporting_details": ["market competitive", f"{location} rates", "internal equity"],
                }
                
                # Run evaluation with enhanced context
                st.markdown("---")
                st.markdown("## 🧠 AI Evaluation Results")
                
                with st.spinner("🔍 Analyzing recommendation quality..."):
                    evaluation = evaluator.evaluate_recommendation(
                        recommendation={"recommendation": workflow_output},
                        context=enhanced_context
                    )
                    
                # Display evaluation dashboard
                create_evaluation_dashboard_ui([evaluation])
                
                # Additional AI-specific evaluation explanation
                with st.expander("📊 About AI Evaluation Dimensions", expanded=True):
                    st.markdown("""
                    ### New AI Evaluation Dimensions
                    
                    This evaluation framework includes advanced AI-specific dimensions:
                    
                    - **Context Relevance**: How well the recommendation relates to the specific role, level, and location context
                    - **Faithfulness**: Whether the recommendation stays true to the data without hallucination
                    - **Context Support Coverage**: How comprehensively the recommendation uses available market data
                    - **Question Answerability**: How directly the recommendation addresses the specific compensation query
                    """)
            
            # Export functionality
            create_export_functionality({
                "workflow_result": workflow_result,
                "context": context,
                "evaluation": evaluation if evaluator else None,
                "timestamp": datetime.now().isoformat()
            }, "compensation_analysis")
            
        else:
            st.error("❌ Multi-Agent workflow encountered issues. Using direct fallback recommendation.")
            fallback = workflow_result.get("fallback_recommendation", {})
            if fallback:
                display_fallback_recommendation(fallback)

def display_rag_interface(co_client, config):
    """Display enhanced RAG interface with dropdowns + semantic extraction + DuckDuckGo fallback"""
    
    st.markdown("### 🔍 RAG-Enhanced Compensation Chat")
    
    # First put the text field (field should be above dropdowns)
    st.text_area(
        "Additional Context",
        placeholder="Any specific requirements, budget constraints, or candidate details...",
        height=100,
        key="rag_additional_context"
    )
    
    st.info("💡 Ask naturally OR use the dropdowns below. I'll search our 150+ real compensation records and use web search as fallback!")
    
    # Initialize RAG system
    rag_system = st.session_state.get("rag_system")
    if not rag_system:
        with st.spinner("🔄 Initializing RAG system with real compensation data..."):
            try:
                from comp_planner.cohere_rag_system import get_cohere_rag_system
                rag_system = get_cohere_rag_system(co_client)
                st.session_state["rag_system"] = rag_system
                st.success("✅ RAG system ready with real compensation database!")
            except Exception as e:
                st.error(f"⚠️ RAG system initialization failed: {e}")
                return
    
    # Get unique values from our real data for dropdowns
    @st.cache_data
    def get_dropdown_options():
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Compensation Data.csv")
            df = pd.read_csv(data_path)
            
            job_titles = sorted(df['job_title'].unique().tolist())
            job_levels = sorted(df['job_level'].unique().tolist())
            locations = sorted(df['location'].unique().tolist())
            
            return job_titles, job_levels, locations
        except Exception:
            # Fallback options
            return (
                ["Software Engineer", "Product Manager", "Data Scientist", "DevOps Engineer"],
                ["Junior", "Mid", "Senior", "Staff", "Principal"],
                ["San Francisco", "New York", "Seattle", "Austin", "Remote"]
            )
    
    job_titles, job_levels, locations = get_dropdown_options()
    
    # Now add the dropdowns after the text field
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_role = st.selectbox(
            "Role/Position",
            options=[""] + job_titles,
            key="rag_role",
            help="Select from actual roles in our database"
        )
    
    with col2:
        selected_level = st.selectbox(
            "Seniority Level", 
            options=[""] + job_levels,
            key="rag_level",
            help="Select from actual levels in our database"
        )
    
    with col3:
        selected_location = st.selectbox(
            "Location",
            options=[""] + locations,
            key="rag_location",
            help="Select from actual locations in our database"
        )
    
    # Add execute button for RAG search (NOT CrewAI workflow)
    if st.button("🔍 Search Compensation Data", type="primary", use_container_width=True):
        if selected_role or selected_level or selected_location:
            # Create a search query from selections
            query_parts = []
            if selected_role:
                query_parts.append(selected_role)
            if selected_level:
                query_parts.append(selected_level)
            if selected_location:
                query_parts.append(f"in {selected_location}")
            
            additional_context = st.session_state.get("rag_additional_context", "")
            dropdown_query = "What is the compensation for " + " ".join(query_parts) + "?"
            
            if additional_context:
                dropdown_query += f" Additional context: {additional_context}"
            
            # Add to chat history and trigger search
            if "rag_messages" not in st.session_state:
                st.session_state.rag_messages = []
            
            st.session_state.rag_messages.append({"role": "user", "content": dropdown_query})
            
            # Execute RAG query (NOT CrewAI workflow)
            with st.spinner("🔍 Searching compensation records..."):
                rag_result = rag_system.generate_compensation_recommendation(
                    query=dropdown_query,
                    role=selected_role,
                    level=selected_level,
                    location=selected_location,
                    search_k=5
                )
                
                if rag_result.get("recommendation"):
                    st.success("✅ Found relevant compensation data!")
                    st.markdown("### 💰 Compensation Recommendation")
                    st.markdown(rag_result["recommendation"])
                    
                    # Show statistics
                    stats = rag_result.get('compensation_stats', {})
                    st.markdown("### 📊 Market Data")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Base", f"${stats.get('avg_base', 0):,.0f}")
                    with col2:
                        st.metric("Average Bonus", f"${stats.get('avg_bonus', 0):,.0f}")
                    with col3:
                        st.metric("Average Total", f"${stats.get('avg_total', 0):,.0f}")
                else:
                    st.error("❌ No relevant compensation data found")
                    st.write(rag_result.get("error", "Try adjusting your search criteria"))
    
    # Display database stats in sidebar
    with st.sidebar:
        st.markdown("### 📊 Database Info")
        db_stats = rag_system.get_database_stats()
        if "error" not in db_stats:
            st.metric("Total Records", db_stats.get("total_records", 0))
            st.metric("Job Titles", db_stats.get("unique_job_titles", 0))
            st.metric("Locations", db_stats.get("unique_locations", 0))
            st.metric("Avg Base Salary", db_stats.get("average_base_salary", "$0"))
            
            # Web fallback toggle
            st.markdown("### 🌐 Fallback Options")
            use_web_fallback = st.checkbox("Enable Web Search Fallback", value=True, 
                                         help="Use DuckDuckGo search when database has no matches")
            st.session_state["use_web_fallback"] = use_web_fallback
        else:
            st.warning("Database loading...")
    
    # Initialize chat history for RAG mode
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
        # Add welcome message
        st.session_state.rag_messages.append({
            "role": "assistant", 
            "content": "👋 Hi! I'm your RAG-enhanced compensation assistant. I can:\n\n• 🔍 **Search 150+ real compensation records** from your database\n• 🌐 **Use web search as fallback** when no matches found\n• 🎯 **Extract info automatically** from natural language\n• 📊 **Use dropdown filters** for precise searches\n\n**Try asking**: *\"What should we pay a Senior Software Engineer in NYC?\"*"
        })
    
    # Display chat history
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Enhanced chat input with auto-extraction
    if prompt := st.chat_input("Ask about compensation (e.g. 'What should we pay a Senior Software Engineer in NYC?')", key="rag_chat_input"):
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate RAG response with web fallback
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching compensation database..."):
                
                # Show RAG process in real-time
                status_container = st.empty()
                status_container.info("🔍 **Step 1**: Analyzing your question...")
                
                # Extract context from the query using Cohere
                try:
                    context_extraction_prompt = f"""
                    Extract compensation-related information from this query: "{prompt}"
                    
                    Identify if mentioned:
                    - Role/Position (e.g., Software Engineer, Product Manager, Data Scientist)
                    - Level (e.g., Junior, Mid, Senior, Staff, Principal, Director)  
                    - Location (e.g., San Francisco, NYC, London, Remote)
                    
                    If any information is missing, use reasonable defaults based on the query context.
                    
                    Respond in this exact format:
                    Role: [extracted role or "Software Engineer"]
                    Level: [extracted level or "Senior"]
                    Location: [extracted location or "San Francisco"]
                    """
                    
                    context_response = co_client.chat(
                        model="command-r-plus",
                        message=context_extraction_prompt,
                        temperature=0.1
                    )
                    
                    # Parse the response to extract role, level, location
                    lines = context_response.text.strip().split('\n')
                    role = "Software Engineer"
                    level = "Senior"
                    location = "San Francisco"
                    
                    for line in lines:
                        if line.startswith("Role:"):
                            role = line.split(":", 1)[1].strip()
                        elif line.startswith("Level:"):
                            level = line.split(":", 1)[1].strip()
                        elif line.startswith("Location:"):
                            location = line.split(":", 1)[1].strip()
                    
                    status_container.info(f"🎯 **Step 2**: Searching for *{level} {role}* in *{location}*...")
                    
                except Exception:
                    # Fallback extraction
                    role = "Software Engineer"
                    level = "Senior" 
                    location = "San Francisco"
                    status_container.info(f"🎯 **Step 2**: Searching with default criteria...")
                
                # Execute RAG query
                rag_result = rag_system.generate_compensation_recommendation(
                    query=prompt,
                    role=role,
                    level=level,
                    location=location,
                    search_k=5
                )
                
                # Check if we got good results or need web fallback
                use_web_fallback = st.session_state.get("use_web_fallback", True)
                benchmark_count = rag_result.get('benchmark_count', 0)
                
                if benchmark_count > 0:
                    status_container.success(f"✅ **Step 3**: Found {benchmark_count} relevant records in database!")
                    use_fallback = False
                else:
                    status_container.warning("⚠️ **Step 3**: No matches in database, checking web fallback...")
                    use_fallback = use_web_fallback
                
                # Display the recommendation or web fallback
                if rag_result.get("recommendation") and "error" not in rag_result and benchmark_count > 0:
                    
                    # Main recommendation from database
                    st.markdown("### 💰 Compensation Recommendation (Database)")
                    st.markdown(rag_result["recommendation"])
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Confidence", f"{rag_result.get('confidence_score', 0):.1f}/10")
                    with col2:
                        st.metric("Data Points", rag_result.get('benchmark_count', 0))
                    with col3:
                        comp_stats = rag_result.get('compensation_stats', {})
                        avg_base = comp_stats.get('avg_base', 0)
                        st.metric("Avg Base Salary", f"${avg_base:,.0f}")
                    with col4:
                        avg_total = comp_stats.get('avg_total', 0)
                        st.metric("Avg Total Comp", f"${avg_total:,.0f}")
                    
                    # Compensation breakdown in expander
                    if 'compensation_stats' in rag_result:
                        with st.expander("📊 Market Data Breakdown", expanded=False):
                            stats = rag_result['compensation_stats']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Base Salary Statistics:**")
                                st.write(f"• Average: ${stats.get('avg_base', 0):,.0f}")
                                st.write(f"• Range: ${stats.get('min_base', 0):,.0f} - ${stats.get('max_base', 0):,.0f}")
                                
                            with col2:
                                st.markdown("**Additional Compensation:**")
                                st.write(f"• Avg Bonus: ${stats.get('avg_bonus', 0):,.0f}")
                                st.write(f"• Avg Equity: ${stats.get('avg_equity', 0):,.0f}")
                    
                    # Sources
                    sources = rag_result.get('sources', [])
                    if sources:
                        with st.expander("📚 Data Sources", expanded=False):
                            for i, source in enumerate(sources[:3], 1):
                                st.write(f"{i}. {source}")
                    
                    # Add to chat history
                    response_text = f"**Compensation Recommendation for {level} {role} in {location}:**\n\n"
                    response_text += rag_result["recommendation"]
                    response_text += f"\n\n*Analysis based on {rag_result['benchmark_count']} real compensation records*"
                    
                    st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
                    
                elif use_fallback:
                    # Use DuckDuckGo fallback
                    status_container.info("🌐 **Step 4**: Searching web for additional insights...")
                    
                    web_result, web_sources = run_duckduckgo_search(f"{level} {role} salary {location}")
                    
                    if web_result and "error" not in web_result.lower():
                        st.markdown("### 🌐 Web Search Results (Fallback)")
                        st.info("No matches found in our database. Here's what I found on the web:")
                        st.markdown(web_result)
                        
                        if web_sources:
                            with st.expander("🔗 Web Sources", expanded=False):
                                for source in web_sources:
                                    st.write(f"• {source}")
                        
                        # Generate AI analysis of web results
                        try:
                            analysis_prompt = f"""
                            Based on this web search data about {level} {role} compensation in {location}, 
                            provide a professional compensation recommendation:
                            
                            Web Data:
                            {web_result}
                            
                            Provide:
                            1. Estimated base salary range
                            2. Bonus expectations
                            3. Equity considerations
                            4. Total compensation estimate
                            5. Market positioning
                            
                            Format as a professional recommendation.
                            """
                            
                            analysis_response = co_client.chat(
                                model="command-r-plus",
                                message=analysis_prompt,
                                temperature=0.7
                            )
                            
                            st.markdown("### 🤖 AI Analysis of Web Data")
                            st.markdown(analysis_response.text)
                            
                            # Add to chat history
                            response_text = f"**Web-based Compensation Analysis for {level} {role} in {location}:**\n\n"
                            response_text += analysis_response.text
                            response_text += f"\n\n*Analysis based on web search results (no database matches found)*"
                            
                        except Exception as e:
                            response_text = f"Found web data for {level} {role} in {location}, but couldn't generate AI analysis: {str(e)}"
                        
                        st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
                        
                    else:
                        st.error("❌ No relevant data found in database or web search")
                        st.session_state.rag_messages.append({
                            "role": "assistant", 
                            "content": f"I couldn't find compensation data for {level} {role} in {location} in our database or through web search. Try a more general query or different role/location."
                        })
                        
                else:
                    error_msg = rag_result.get("error", "Failed to generate recommendation")
                    st.error(f"❌ {error_msg}")
                    st.session_state.rag_messages.append({"role": "assistant", "content": f"I encountered an issue: {error_msg}"})
    
    # Quick example queries
    st.markdown("### 💡 Try These Examples:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Senior SWE in NYC", use_container_width=True):
            example_query = "What should we pay a Senior Software Engineer in New York?"
            st.session_state.rag_messages.append({"role": "user", "content": example_query})
            st.rerun()
    
    with col2:
        if st.button("Product Manager SF", use_container_width=True):
            example_query = "Show me Product Manager compensation in San Francisco"
            st.session_state.rag_messages.append({"role": "user", "content": example_query})
            st.rerun()
    
    with col3:
        if st.button("Data Scientist Range", use_container_width=True):
            example_query = "What's the salary range for Data Scientists?"
            st.session_state.rag_messages.append({"role": "user", "content": example_query})
            st.rerun()

def display_evaluation_interface(co_client, config):
    """Display evaluation and analysis interface"""
    
    st.markdown("### 📊 Evaluation & Analysis Dashboard")
    
    # Load evaluation history
    evaluator = st.session_state.get("evaluator")
    consensus_analyzer = st.session_state.get("consensus_analyzer")
    
    if evaluator and evaluator.evaluation_history:
        create_evaluation_dashboard_ui(
            evaluator.evaluation_history,
            consensus_analyzer.consensus_history if consensus_analyzer else None
        )
    else:
        st.info("No evaluation data available yet. Run some compensation queries first!")
        
        # Manual evaluation interface
        st.markdown("### 🧪 Manual Evaluation")
        
        recommendation_text = st.text_area(
            "Paste a compensation recommendation to evaluate:",
            height=200,
            placeholder="Paste the compensation recommendation text here..."
        )
        
        if st.button("📊 Evaluate Recommendation") and recommendation_text:
            evaluation = evaluator.evaluate_recommendation(
                recommendation={"recommendation": recommendation_text}
            )
            
            create_evaluation_dashboard_ui([evaluation])

def display_single_agent_interface(co_client, config):
    """Display single agent interface using the offer_chain approach"""
    
    st.markdown("### 🤖 Single Agent Mode")
    st.info("Using the offer_chain system for compensation planning")
    
    # Initialize session state for chat history
    if "comp_messages" not in st.session_state:
        st.session_state.comp_messages = []
    
    # Display chat history
    for message in st.session_state.comp_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What compensation question can I help you with?"):
        # Add user message to chat history
        st.session_state.comp_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                # Use offer_chain system instead of agent_planner
                try:
                    # Import from agents.offer_chain
                    from agents.offer_chain import run_compensation_planner
                    
                    # Execute the offer_chain workflow
                    result = run_compensation_planner(
                        user_query=prompt,
                        cohere_key=st.session_state.get("cohere_api_key", "")
                    )
                    
                    # Extract the recommendation from recruitment manager
                    if result and "recruitment_manager" in result:
                        offer = result["recruitment_manager"].get("offer", "")
                        st.markdown("### 💰 Compensation Recommendation")
                        st.markdown(offer)
                        
                        # Show HR Director feedback
                        if "hr_director" in result:
                            hr_comments = result["hr_director"].get("comments", "")
                            with st.expander("💼 HR Director Feedback", expanded=False):
                                st.markdown(hr_comments)
                        
                        # Show Hiring Manager approval
                        if "hiring_manager" in result:
                            approval = result["hiring_manager"].get("approval", "Pending")
                            manager_comments = result["hiring_manager"].get("comments", "")
                            
                            status_color = "green" if approval == "Approved" else "red"
                            st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold;'>{approval}</span>", unsafe_allow_html=True)
                            
                            with st.expander("👔 Hiring Manager Comments", expanded=False):
                                st.markdown(manager_comments)
                        
                        # Add to chat history
                        response_content = f"### Compensation Recommendation\n\n{offer}\n\n"
                        if hr_comments:
                            response_content += f"**HR Director Comments:** {hr_comments}\n\n"
                        if approval:
                            response_content += f"**Status:** {approval}"
                        
                        st.session_state.comp_messages.append({"role": "assistant", "content": response_content})
                    else:
                        st.error("Failed to generate compensation recommendation")
                        st.session_state.comp_messages.append({"role": "assistant", "content": "I encountered an error generating your compensation recommendation. Please try again."})
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.comp_messages.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})
                    
    # Quick examples
    if not st.session_state.comp_messages:
        st.markdown("### 💡 Try These Examples:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create a compensation package for a Senior SWE"):
                st.session_state.comp_messages.append({"role": "user", "content": "Create a compensation package for a Senior Software Engineer in San Francisco"})
                st.experimental_rerun()
        with col2:
            if st.button("What's a fair salary for a Product Manager?"):
                st.session_state.comp_messages.append({"role": "user", "content": "What's a fair salary for a Product Manager in New York?"})
                st.experimental_rerun()

def display_multi_agent_interface(co_client, config):
    """Display Multi-Agent interface using fallback system (no CrewAI dependency)"""
    
    st.markdown("### 🤖 Multi-Agent Compensation Planning")
    st.info("This mode simulates three specialized agents: Recruitment Manager → HR Director → Hiring Manager using AI-powered fallback system")
    
    # Input form for compensation request
    with st.form("multi_agent_request"):
        st.markdown("**Compensation Request Details:**")
        # Move text input to the top
        query_text = st.text_area(
            "Describe your compensation request (required)",
            value=st.session_state.get("multi_agent_query_text", ""),
            placeholder="e.g. Create a compensation package for a Junior Software Engineer in San Francisco",
            key="multi_agent_query_text"
        )
        st.markdown("---")
        st.markdown("**Optional: Fill in details below or leave blank**")
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("Role/Position", value="", help="e.g., Software Engineer, Product Manager", key="multi_agent_role")
            level = st.selectbox("Seniority Level", ["", "Junior", "Mid", "Senior", "Staff", "Principal", "Director"], key="multi_agent_level")
        with col2:
            location = st.text_input("Location", value="", help="City or Remote", key="multi_agent_location")
            department = st.text_input("Department", value="", help="Department or team", key="multi_agent_department")
        submitted = st.form_submit_button("🚀 Execute Multi-Agent Workflow")
    
    if submitted and query_text:
        # Create context from form inputs or extract from query
        context = {
            "role": role or "Software Engineer",
            "level": level or "Junior", 
            "location": location or "San Francisco",
            "department": department or "Engineering",
            "query": query_text
        }
        
        # Execute multi-agent workflow using fallback system
        workflow_result = execute_multi_agent_workflow_fallback(query_text, context, co_client)
        
        # Display workflow progress
        from comp_planner.enhanced_ui import display_agent_workflow_progress
        display_agent_workflow_progress(workflow_result)
        
        if workflow_result.get("success", False):
            st.success("🎉 Multi-Agent workflow completed successfully!")
            
            # Display the final recommendation
            st.markdown("### 📋 Final Compensation Recommendation")
            workflow_output = workflow_result.get("workflow_result", "")
            if workflow_output:
                st.markdown(workflow_output)
            
            # Display individual agent outputs
            if "agent_outputs" in workflow_result:
                with st.expander("🔍 View Individual Agent Analysis", expanded=False):
                    agent_outputs = workflow_result["agent_outputs"]
                    
                    tab1, tab2, tab3 = st.tabs(["👤 Recruitment Manager", "🏢 HR Director", "👔 Hiring Manager"])
                    
                    with tab1:
                        st.markdown("**Draft Compensation Package:**")
                        st.markdown(agent_outputs.get("recruitment_manager", "No output available"))
                    
                    with tab2:
                        st.markdown("**Policy Validation & Review:**")
                        st.markdown(agent_outputs.get("hr_director", "No output available"))
                    
                    with tab3:
                        st.markdown("**Final Approval Decision:**")
                        st.markdown(agent_outputs.get("hiring_manager", "No output available"))
            
            # Auto-evaluation if enabled
            evaluator = st.session_state.get("evaluator")
            if evaluator:
                # Pass enhanced context for new AI dimensions
                enhanced_context = {
                    **context,
                    "key_points": [f"{level} {role}", location, department],
                    "supporting_details": ["market competitive", f"{location} rates", "internal equity"],
                }
                
                # Run evaluation with enhanced context
                st.markdown("---")
                st.markdown("## 🧠 AI Evaluation Results")
                
                with st.spinner("🔍 Analyzing recommendation quality..."):
                    evaluation = evaluator.evaluate_recommendation(
                        recommendation={"recommendation": workflow_output},
                        context=enhanced_context
                    )
                    
                # Display evaluation dashboard
                create_evaluation_dashboard_ui([evaluation])
                
                # Additional AI-specific evaluation explanation
                with st.expander("📊 About AI Evaluation Dimensions", expanded=True):
                    st.markdown("""
                    ### New AI Evaluation Dimensions
                    
                    This evaluation framework includes advanced AI-specific dimensions:
                    
                    - **Context Relevance**: How well the recommendation relates to the specific role, level, and location context
                    - **Faithfulness**: Whether the recommendation stays true to the data without hallucination
                    - **Context Support Coverage**: How comprehensively the recommendation uses available market data
                    - **Question Answerability**: How directly the recommendation addresses the specific compensation query
                    """)
            
            # Export functionality
            create_export_functionality({
                "workflow_result": workflow_result,
                "context": context,
                "evaluation": evaluation if evaluator else None,
                "timestamp": datetime.now().isoformat()
            }, "compensation_analysis")
            
        else:
            st.error("❌ Multi-Agent workflow encountered issues. Using direct fallback recommendation.")
            fallback = workflow_result.get("fallback_recommendation", {})
            if fallback:
                display_fallback_recommendation(fallback)

def execute_multi_agent_workflow_fallback(query: str, context: Dict[str, Any], co_client) -> Dict[str, Any]:
    """Execute multi-agent workflow using Cohere-powered fallback (no CrewAI)"""
    
    start_time = datetime.now()
    
    try:
        # Get market data using the tool
        from comp_planner.crewai_agents import MarketResearchTool
        market_tool = MarketResearchTool()
        market_data = market_tool._run(context["role"], context["level"], context["location"])
        
        # Step 1: Recruitment Manager - Draft initial offer
        recruitment_prompt = f"""
        You are an experienced Recruitment Manager tasked with drafting a competitive compensation offer.
        
        Request: {query}
        Role: {context['level']} {context['role']}
        Location: {context['location']}
        Department: {context['department']}
        
        Market Data:
        - Base Salary Range: ${market_data['base_salary_range']['min']:,} - ${market_data['base_salary_range']['max']:,}
        - Market P50: ${market_data['market_percentiles']['p50']:,}
        - Average Bonus: ${market_data['bonus_statistics']['avg_amount']:,} ({market_data['bonus_statistics']['avg_percentage']}%)
        - Average Equity: ${market_data['equity_statistics']['avg_amount']:,} ({market_data['equity_statistics']['avg_percentage']}%)
        
        As a Recruitment Manager, draft a competitive compensation package that:
        1. Attracts top talent
        2. Is market-competitive
        3. Includes base salary, bonus, equity, and benefits
        4. Provides clear justification for each component
        
        Format as a professional draft offer.
        """
        
        recruitment_response = co_client.chat(
            model="command-r-plus",
            message=recruitment_prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Step 2: HR Director - Policy validation and review
        hr_prompt = f"""
        You are an HR Director reviewing the following compensation package for policy compliance and internal equity.
        
        Original Request: {query}
        
        Recruitment Manager's Draft:
        {recruitment_response.text}
        
        Market Context:
        - Sample size: {market_data.get('sample_size', 0)} records
        - Market confidence: {market_data.get('confidence', 0):.1%}
        
        As an HR Director, please:
        1. Validate policy compliance
        2. Check for internal equity considerations
        3. Assess long-term sustainability
        4. Recommend any necessary adjustments
        5. Identify required approval levels
        
        Provide your professional HR assessment and any modifications needed.
        """
        
        hr_response = co_client.chat(
            model="command-r-plus", 
            message=hr_prompt,
            temperature=0.6,
            max_tokens=800
        )
        
        # Step 3: Hiring Manager - Final approval decision
        hiring_manager_prompt = f"""
        You are a Hiring Manager making the final decision on this compensation package.
        
        Original Request: {query}
        Budget Context: Department - {context['department']}, Location - {context['location']}
        
        Recruitment Manager's Draft:
        {recruitment_response.text}
        
        HR Director's Review:
        {hr_response.text}
        
        As the Hiring Manager, please:
        1. Consider team budget constraints
        2. Evaluate business impact and ROI
        3. Make final approval decision (Approved/Needs Revision/Rejected)
        4. Provide final adjustments if needed
        5. Include any conditions or contingencies
        
        Give your final decision with clear business justification.
        """
        
        hiring_manager_response = co_client.chat(
            model="command-r-plus",
            message=hiring_manager_prompt, 
            temperature=0.6,
            max_tokens=800
        )
        
        # Compile final recommendation
        final_recommendation = f"""
        ## Multi-Agent Compensation Planning Result
        
        **Final Approved Package for {context['level']} {context['role']} in {context['location']}**
        
        ### Executive Summary
        After thorough analysis by our three-agent team, here is the final compensation recommendation:
        
        {hiring_manager_response.text}
        
        ---
        
        *This recommendation was developed through a comprehensive three-stage process involving market research, policy validation, and business case evaluation.*
        """
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "workflow_result": final_recommendation,
            "execution_time": execution_time,
            "tasks_completed": 3,
            "agents_involved": ["Recruitment Manager", "HR Director", "Hiring Manager"],
            "agent_outputs": {
                "recruitment_manager": recruitment_response.text,
                "hr_director": hr_response.text, 
                "hiring_manager": hiring_manager_response.text
            },
            "market_data": market_data,
            "context": context
        }
        
    except Exception as e:
        # Fallback to simple recommendation
        return {
            "success": False,
            "error": str(e),
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "fallback_recommendation": "Unable to generate recommendation due to system error.",
            "context": context
        }

def display_fallback_recommendation(fallback_data: Dict[str, Any]):
    """Display fallback recommendation when workflow fails"""
    
    st.markdown("### 💡 Fallback Compensation Recommendation")
    st.info("Workflow encountered issues. Here's a data-driven recommendation using our fallback system:")
    
    if isinstance(fallback_data, dict):
        # Display main recommendation text
        if "recommendation" in fallback_data:
            st.markdown("#### 📋 Recommendation")
            st.markdown(fallback_data["recommendation"])
        
        # Display structured compensation data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if "base_salary" in fallback_data:
                salary_info = fallback_data["base_salary"]
                if isinstance(salary_info, dict):
                    recommended = salary_info.get("recommended", 0)
                    st.metric("Base Salary", f"${recommended:,}")
                    
                    # Show range if available
                    salary_range = salary_info.get("range", {})
                    if salary_range:
                        min_sal = salary_range.get("min", 0)
                        max_sal = salary_range.get("max", 0)
                        st.caption(f"Range: ${min_sal:,} - ${max_sal:,}")
        
        with col2:
            if "bonus" in fallback_data:
                bonus_info = fallback_data["bonus"]
                if isinstance(bonus_info, dict):
                    target_bonus = bonus_info.get("target", 0)
                    st.metric("Annual Bonus", f"${target_bonus:,}")
                    
                    # Show percentage if available
                    percentage = bonus_info.get("percentage", 0)
                    if percentage:
                        st.caption(f"{percentage:.1f}% of base")
        
        with col3:
            if "equity" in fallback_data:
                equity_info = fallback_data["equity"]
                if isinstance(equity_info, dict):
                    equity_value = equity_info.get("estimated_value", 0)
                    st.metric("Equity Value", f"${equity_value:,}")
                    
                    # Show percentage if available
                    equity_pct = equity_info.get("percentage", 0)
                    if equity_pct:
                        st.caption(f"{equity_pct:.3f}% equity")
        
        # Total compensation summary
        if "total_compensation" in fallback_data:
            total_comp = fallback_data["total_compensation"]
            if isinstance(total_comp, dict):
                estimated_total = total_comp.get("estimated", 0)
                st.markdown("#### 💰 Total Compensation Package")
                st.success(f"**Estimated Total: ${estimated_total:,}**")
        
        # Show confidence and source
        col1, col2 = st.columns(2)
        with col1:
            confidence = fallback_data.get("confidence_score", 0)
            st.metric("Confidence Score", f"{confidence:.1f}/10")
        
        with col2:
            source = fallback_data.get("source", "unknown")
            st.metric("Data Source", source.replace("_", " ").title())
        
        # Market data details in expander
        if "market_data" in fallback_data:
            with st.expander("📊 Market Data Details", expanded=False):
                market_data = fallback_data["market_data"]
                
                if "sample_size" in market_data:
                    st.write(f"**Sample Size**: {market_data['sample_size']} records")
                
                if "base_salary_range" in market_data:
                    salary_range = market_data["base_salary_range"]
                    st.write(f"**Market Salary Range**: ${salary_range.get('min', 0):,} - ${salary_range.get('max', 0):,}")
                    st.write(f"**Market Average**: ${salary_range.get('avg', 0):,}")
                
                if "market_percentiles" in market_data:
                    percentiles = market_data["market_percentiles"]
                    st.write("**Market Percentiles**:")
                    st.write(f"- 25th: ${percentiles.get('p25', 0):,}")
                    st.write(f"- 50th: ${percentiles.get('p50', 0):,}")
                    st.write(f"- 75th: ${percentiles.get('p75', 0):,}")
                    st.write(f"- 90th: ${percentiles.get('p90', 0):,}")
        
        # Error information if available
        if "error" in fallback_data:
            with st.expander("⚠️ Error Details", expanded=False):
                st.error(f"Original error: {fallback_data['error']}")
    
    else:
        # Simple fallback for non-dict data
        st.write(str(fallback_data))
    
    st.warning("💡 **Note**: This is a fallback recommendation. Please check system dependencies and try again.")

if __name__ == "__main__":
    # Initialize Cohere client first
    if 'cohere_client' not in st.session_state:
        cohere_client = initialize_cohere_client()
        if cohere_client:
            st.session_state.cohere_client = cohere_client
            st.session_state.cohere_api_key = get_api_keys()
    
    # Run the main app
    if 'cohere_client' in st.session_state:
        main()
    else:
        st.error("Failed to initialize Cohere client. Please check your API key configuration.")
