"""
Enhanced Compensation Planner Application with CrewAI Multi-Agent System.
Integrates all 4 phases: CrewAI agents, Cohere RAG, evaluation framework, and enhanced UI.
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

# Try to import CrewAI components
try:
    from comp_planner.crewai_agents import get_crewai_compensation_planner, CompensationContext
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Fallback class if CrewAI not available
    class CompensationContext:
        def __init__(self, role="Software Engineer", level="Junior", location="San Francisco", department="Engineering"):
            self.role = role
            self.level = level
            self.location = location
            self.department = department
        
        def dict(self):
            return {
                "role": self.role,
                "level": self.level,
                "location": self.location,
                "department": self.department
            }

from comp_planner.evaluation_framework import get_compensation_evaluator, get_agent_consensus_analyzer
from comp_planner.enhanced_ui import (
    display_enhanced_sidebar, create_evaluation_dashboard_ui, 
    display_agent_workflow_progress, display_recommendation_comparison,
    display_workflow_metrics, create_export_functionality
)

# Import existing modules
from . import auth
from . import query_router
# Import offer_chain instead of agent_planner
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
        # Initialize CrewAI system
        systems["crewai"] = get_crewai_compensation_planner(cohere_client)
        st.session_state["crewai_system"] = systems["crewai"]
        
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

def execute_crewai_workflow(query: str, context: CompensationContext, crewai_system) -> Dict[str, Any]:
    """Execute the CrewAI multi-agent workflow"""
    
    with st.spinner("🤖 Running CrewAI Multi-Agent Workflow..."):
        
        # Show workflow progress
        workflow_placeholder = st.empty()
        
        with workflow_placeholder.container():
            st.markdown("### 🔄 Agent Workflow in Progress")
            
            # Show initial status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/3: Recruitment Manager drafting offer...")
            progress_bar.progress(33)
            
            # Execute the workflow
            start_time = datetime.now()
            result = crewai_system.execute_compensation_planning(context)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            if result.get("success", False):
                status_text.text("Step 2/3: HR Director validating policy...")
                progress_bar.progress(66)
                
                status_text.text("Step 3/3: Hiring Manager approving...")
                progress_bar.progress(100)
                
                status_text.text("✅ Workflow completed successfully!")
                
                # Add execution metrics
                result["execution_time"] = execution_time
                result["workflow_steps"] = 3
                
            else:
                status_text.text("❌ Workflow encountered issues, using fallback...")
                progress_bar.progress(100)
        
        # Clear the progress display
        workflow_placeholder.empty()
        
        return result

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
    """Enhanced main function with all 4 phases integrated"""
    
    # Initialize Cohere client
    if 'cohere_client' not in st.session_state:
        st.error("Cohere client not initialized. Please go back and configure API key.")
        return
    
    co_client = st.session_state.cohere_client
    
    # Initialize enhanced systems
    if 'systems_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing enhanced compensation planning systems..."):
            systems = initialize_enhanced_systems(co_client)
            st.session_state['systems_initialized'] = True
            st.success("✅ All systems initialized successfully!")
    
    # Enhanced sidebar with advanced controls
    sidebar_config = display_enhanced_sidebar()
    
    # App header
    st.title("💰 Enhanced Compensation Planner")
    st.caption("🤖 CrewAI Multi-Agent System • 🔍 Cohere RAG • 📊 Evaluation Framework")
    
    # Mode-specific interface
    agent_mode = sidebar_config.get("agent_mode", "CrewAI Multi-Agent")
    
    if agent_mode == "CrewAI Multi-Agent":
        display_crewai_interface(co_client, sidebar_config)
    elif agent_mode == "RAG Enhanced":
        display_rag_interface(co_client, sidebar_config)
    elif agent_mode == "Evaluation Mode":
        display_evaluation_interface(co_client, sidebar_config)
    else:
        display_single_agent_interface(co_client, sidebar_config)

def display_crewai_interface(co_client, config):
    """Display CrewAI multi-agent interface"""
    
    st.markdown("### 🤖 CrewAI Multi-Agent Compensation Planning")
    st.info("This mode uses three specialized agents: Recruitment Manager → HR Director → Hiring Manager")
    
    # Input form for compensation request
    with st.form("crewai_request"):
        st.markdown("**Compensation Request Details:**")
        # Move text input to the top
        query_text = st.text_area(
            "Describe your compensation request (required)",
            placeholder="e.g. Create a compensation package for a Junior Software Engineer in San Francisco",
            key="crewai_query_text"
        )
        st.markdown("---")
        st.markdown("**Optional: Fill in details below or leave blank**")
        col1, col2 = st.columns(2)
        with col1:
            role = st.text_input("Role/Position", value="", help="e.g., Software Engineer, Product Manager", key="crewai_role")
            level = st.selectbox("Seniority Level", ["", "Junior", "Mid", "Senior", "Staff", "Principal", "Director"], key="crewai_level")
        with col2:
            location = st.text_input("Location", value="", help="City or Remote", key="crewai_location")
            department = st.text_input("Department", value="", help="Department or team", key="crewai_department")
        submitted = st.form_submit_button("🚀 Execute CrewAI Workflow")
    
    if submitted and query_text:
        # Use dropdowns if filled, else try to extract from query_text
        context = CompensationContext(
            role=role or "Software Engineer",
            level=level or "Junior",
            location=location or "San Francisco",
            department=department or "Engineering"
        )
        crewai_system = st.session_state.get("crewai_system")
        if crewai_system:
            workflow_result = execute_crewai_workflow(
                query=query_text,
                context=context,
                crewai_system=crewai_system
            )
            display_agent_workflow_progress(workflow_result)
            if workflow_result.get("success", False):
                st.success("🎉 CrewAI workflow completed successfully!")
                display_workflow_metrics({
                    "execution_time": workflow_result.get("execution_time", 0),
                    "tasks_completed": workflow_result.get("tasks_completed", 3),
                    "agents_involved": workflow_result.get("agents_involved", []),
                    "success": True
                })
                st.markdown("### 📋 Final Compensation Recommendation")
                workflow_output = workflow_result.get("workflow_result", "")
                if workflow_output:
                    st.markdown(workflow_output)
                else:
                    fallback = workflow_result.get("fallback_recommendation", {})
                    if fallback:
                        display_fallback_recommendation(fallback)
                if config.get("auto_eval", False):
                    evaluator = st.session_state.get("evaluator")
                    if evaluator:
                        evaluation = evaluator.evaluate_recommendation(
                            recommendation={"recommendation": workflow_output or fallback},
                            context={"role": context.role, "level": context.level, "location": context.location}
                        )
                        st.markdown("### 📊 Automated Evaluation")
                        create_evaluation_dashboard_ui([evaluation])
                create_export_functionality({
                    "workflow_result": workflow_result,
                    "context": context.dict(),
                    "timestamp": datetime.now().isoformat()
                }, "crewai_workflow")
            else:
                st.error("❌ CrewAI workflow failed. Using fallback recommendation.")
                fallback = workflow_result.get("fallback_recommendation", {})
                if fallback:
                    display_fallback_recommendation(fallback)
        else:
            st.error("CrewAI system not available")

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
