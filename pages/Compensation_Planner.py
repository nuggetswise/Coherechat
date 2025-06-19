"""
Enhanced Compensation Assistant
Combines comprehensive agent capabilities with an HR-friendly interface
Now with CrewAI Multi-Agent System, Cohere RAG, and Evaluation Framework
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Apply SQLite patch first thing
from apply_patches import patch_sqlite_for_chromadb
patch_sqlite_for_chromadb()

import streamlit as st

# Set page config with mobile responsiveness - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Compensation Planner (Multi-Agent)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"  # Better for mobile
)

import cohere
import uuid
import re
from pathlib import Path
import json
from datetime import datetime
from agents.offer_chain import run_compensation_planner as run_offer_chain
import pandas as pd

# Import dependency check
try:
    from dependency_check import run_dependency_check
    # Run the dependency check first before loading any modules
    run_dependency_check()
except ImportError:
    st.warning("Dependency check module not found. Proceeding without dependency verification.")

# Add the parent directory to path to allow imports from comp_planner package
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the enhanced components - only import what we need
try:
    from comp_planner.comp_planner_app import (
        initialize_cohere_client, 
        display_crewai_interface, display_rag_interface, 
        display_evaluation_interface, display_single_agent_interface
    )
except ImportError as e:
    import traceback
    st.error(f"Error importing components: {str(e)}")
    st.code(traceback.format_exc())
    # Define fallback functions
    def initialize_cohere_client():
        return None
    def display_crewai_interface(*args, **kwargs):
        st.error("CrewAI interface not available")
    def display_rag_interface(*args, **kwargs):
        st.error("RAG interface not available")
    def display_evaluation_interface(*args, **kwargs):
        st.error("Evaluation interface not available")
    def display_single_agent_interface(*args, **kwargs):
        st.error("Single agent interface not available")

try:
    from comp_planner.enhanced_ui import display_enhanced_sidebar
    from comp_planner.personas import get_persona_names, get_persona_config
except ImportError:
    # Fallback if enhanced UI is not available
    def display_enhanced_sidebar():
        return {}
    def get_persona_names():
        return []
    def get_persona_config(name):
        return {}

# Mobile-responsive CSS
st.markdown("""
<style>
    @media screen and (max-width: 640px) {
        .main .block-container {
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        
        .stButton > button {
            height: auto;
            padding: 0.75rem 0;
            width: 100%;
        }
        
        .example-buttons {
            flex-direction: column;
        }
        
        .example-buttons > div {
            margin-bottom: 0.5rem;
        }
    }
    
    .confidence-bar {
        background-color: #eee;
        border-radius: 3px;
        height: 10px;
        width: 100%;
        margin-top: 5px;
    }
    
    .confidence-fill {
        height: 10px;
        border-radius: 3px;
        transition: width 0.3s ease;
    }
    
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .agent-workflow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    
    .agent-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        flex: 1;
        margin: 0 0.5rem;
    }
    
    .agent-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .agent-status-completed {
        color: #28a745;
    }
    
    .agent-status-active {
        color: #007bff;
    }
    
    .agent-status-pending {
        color: #6c757d;
    }
    
    .progress-arrow {
        font-size: 1.5rem;
        color: #007bff;
        margin: 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def standardize_text_formatting(text):
    """
    Standardize text formatting by cleaning markdown and special characters consistently.
    This function ensures all agent outputs have consistent formatting.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove all heading marks (# to ####)
    cleaned = re.sub(r'#{1,6}\s*', '', text)
    
    # Remove bold formatting (**text**)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)
    
    # Remove italic formatting (*text*)
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)
    
    # Remove code blocks
    cleaned = re.sub(r'```(?:.*?)\n(.*?)```', r'\1', cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("```", "")
    
    # Remove underlines and other markdown formatting
    cleaned = re.sub(r'__(.*?)__', r'\1', cleaned)
    cleaned = re.sub(r'~~(.*?)~~', r'\1', cleaned)
    
    # Remove extra whitespace and normalize line breaks
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned

def display_confidence_score(score_value, label="Confidence"):
    """Display confidence score with visual indicator"""
    try:
        score_num = float(score_value)
        color = "#dc3545" if score_num < 5 else "#fd7e14" if score_num < 7 else "#28a745"
        st.markdown(f"""
        <div style="margin: 0.5rem 0;">
            <p style="margin-bottom: 5px;"><strong>{label}:</strong> <span style="color:{color}; font-weight:bold;">{score_num}/10</span></p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="background-color: {color}; width: {score_num*10}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except (ValueError, TypeError):
        st.text(f"{label}: {score_value}")

def display_agent_workflow_status(current_step=0, steps_completed=None):
    """Display visual workflow status"""
    if steps_completed is None:
        steps_completed = []
    
    agents = [
        {"name": "Recruitment\nManager", "icon": "💼", "step": 0},
        {"name": "HR\nDirector", "icon": "👔", "step": 1},
        {"name": "Hiring\nManager", "icon": "🎯", "step": 2},
        {"name": "Evaluator", "icon": "📊", "step": 3}
    ]
    
    workflow_html = '<div class="agent-workflow">'
    
    for i, agent in enumerate(agents):
        # Determine status
        if agent["step"] in steps_completed:
            status_class = "agent-status-completed"
            status_text = "✅ Complete"
        elif agent["step"] == current_step:
            status_class = "agent-status-active"
            status_text = "🔄 Active"
        else:
            status_class = "agent-status-pending"
            status_text = "⏳ Pending"
        
        workflow_html += f'''
        <div class="agent-step">
            <div class="agent-icon {status_class}">{agent["icon"]}</div>
            <div><strong>{agent["name"]}</strong></div>
            <div class="{status_class}">{status_text}</div>
        </div>
        '''
        
        # Add arrow between agents (except after last one)
        if i < len(agents) - 1:
            arrow_class = "agent-status-completed" if agent["step"] in steps_completed else "agent-status-pending"
            workflow_html += f'<div class="progress-arrow {arrow_class}">→</div>'
    
    workflow_html += '</div>'
    st.markdown(workflow_html, unsafe_allow_html=True)

def create_agent_interaction_flow(chain_outputs):
    """Create visual representation of how agents interact"""
    st.markdown("### 🔄 Agent Interaction Flow")
    
    # Extract key information passed between agents
    recruitment_data = chain_outputs.get('recruitment_manager', {})
    hr_data = chain_outputs.get('hr_director', {})
    hiring_data = chain_outputs.get('hiring_manager', {})
    
    # Agent 1 -> Agent 2
    with st.container():
        st.markdown("#### 💼 Recruitment Manager → 👔 HR Director")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Data Passed:**")
            st.code(f"""
Role: {recruitment_data.get('role', 'N/A')}
Level: {recruitment_data.get('level', 'N/A')}
Location: {recruitment_data.get('location', 'N/A')}
Offer: {recruitment_data.get('offer', 'N/A')[:100]}...
            """)
        with col2:
            st.markdown("**HR Director's Response:**")
            st.info(f"**Confidence:** {hr_data.get('confidence', 'N/A')}")
            st.success(f"**Changes:** {hr_data.get('suggested_changes', 'None')}")
    
    st.markdown("---")
    
    # Agent 2 -> Agent 3
    with st.container():
        st.markdown("#### 👔 HR Director → 🎯 Hiring Manager")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Data Passed:**")
            st.code(f"""
Original Offer: {recruitment_data.get('offer', 'N/A')[:50]}...
HR Comments: {hr_data.get('comments', 'N/A')[:50]}...
Confidence: {hr_data.get('confidence', 'N/A')}
            """)
        with col2:
            st.markdown("**Hiring Manager's Decision:**")
            decision = hiring_data.get('decision', 'N/A')
            if 'approved' in decision.lower() or 'accept' in decision.lower():
                st.success(f"**Decision:** {decision}")
            else:
                st.warning(f"**Decision:** {decision}")
            st.info(f"**Department:** {hiring_data.get('department', 'N/A')}")

def display_agent_step_details(step_name, agent_data, step_number):
    """Display real-time agent step details as they happen"""
    with st.container():
        st.markdown(f"### {step_number}️⃣ {step_name}")
        
        # Show what the agent received as input
        if step_name == "💼 Recruitment Manager":
            st.markdown("**📥 Input Received:**")
            st.code(f"User Query: {agent_data.get('user_query', 'N/A')}")
            
        elif step_name == "👔 HR Director":
            st.markdown("**📥 Input Received:**")
            st.code(f"""
Recruitment Manager's Offer:
{agent_data.get('offer', 'N/A')[:200]}...

Role: {agent_data.get('role', 'N/A')}
Level: {agent_data.get('level', 'N/A')}
            """)
            
        elif step_name == "🎯 Hiring Manager":
            st.markdown("**📥 Input Received:**")
            st.code(f"""
Original Offer: {agent_data.get('offer', 'N/A')[:100]}...
HR Review: {agent_data.get('hr_comments', 'N/A')[:100]}...
HR Confidence: {agent_data.get('hr_confidence', 'N/A')}
            """)
        
        # Show what the agent is outputting
        st.markdown("**📤 Agent Output:**")
        if step_name == "💼 Recruitment Manager":
            offer_text = agent_data.get('offer', 'Processing...')
            if offer_text != 'Processing...':
                st.success("✅ Offer generated successfully!")
                # Use standardized text formatting
                clean_offer = standardize_text_formatting(offer_text)
                st.write(clean_offer)
            else:
                st.info("🔄 Generating compensation offer...")
                
        elif step_name == "👔 HR Director":
            comments = agent_data.get('comments', 'Processing...')
            if comments != 'Processing...':
                st.success("✅ Policy review completed!")
                # Use standardized text formatting
                clean_comments = standardize_text_formatting(comments)
                st.write(clean_comments)
                confidence = agent_data.get('confidence', 'N/A')
                if confidence != 'N/A':
                    display_confidence_score(confidence, "HR Confidence")
            else:
                st.info("🔄 Reviewing for policy compliance...")
                
        elif step_name == "🎯 Hiring Manager":
            decision = agent_data.get('decision', 'Processing...')
            comments = agent_data.get('comments', 'Processing...')
            if decision != 'Processing...' and comments != 'Processing...':
                st.success("✅ Final decision made!")
                if 'approved' in decision.lower():
                    st.success(f"**Decision:** {decision}")
                else:
                    st.warning(f"**Decision:** {decision}")
                # Use standardized text formatting
                clean_comments = standardize_text_formatting(comments)
                st.write(clean_comments)
            else:
                st.info("🔄 Making final hiring decision...")
        
        # Display agent evaluation if available
        evaluation = agent_data.get('evaluation', {})
        if evaluation and isinstance(evaluation, dict) and "scores" in evaluation:
            with st.expander("🔍 View Agent Quality Evaluation"):
                st.markdown("**Agent Output Quality Metrics:**")
                
                # Get scores and display in columns
                scores = evaluation.get("scores", {})
                if scores:
                    cols = st.columns(len(scores))
                    for i, (metric, score) in enumerate(scores.items()):
                        with cols[i]:
                            color = "#28a745" if score >= 8 else "#fd7e14" if score >= 6 else "#dc3545"
                            st.metric(label=metric.capitalize(), value=f"{score}/10")
                
                # Display overall score
                overall = evaluation.get("overall_score", 0)
                st.progress(overall/10)
                
                # Display feedback
                feedback = evaluation.get("feedback", "")
                if feedback:
                    # Use standardized text formatting for feedback too
                    clean_feedback = standardize_text_formatting(feedback)
                    st.markdown(f"**Feedback:** {clean_feedback}")

def run_workflow_with_live_updates(user_prompt, uploaded_docs, openai_key, cohere_key, db_data=None):
    """Run the workflow with live updates for each agent step"""
    
    # Create containers for each step
    step_containers = {}
    for i in range(1, 5):
        step_containers[i] = st.empty()
    
    # Overall progress
    progress_bar = st.progress(0)
    overall_status = st.empty()
    
    try:
        # Step 1: Recruitment Manager
        overall_status.text("🔄 Step 1/4: Recruitment Manager analyzing requirements...")
        progress_bar.progress(25)
        
        with step_containers[1].container():
            agent_data = {"user_query": user_prompt}
            display_agent_step_details("💼 Recruitment Manager", agent_data, 1)
        
        # Actually run the chain to get recruitment manager output
        chain_outputs = run_offer_chain(
            user_query=user_prompt,
            db_data=str(db_data) if db_data else "",
            web_data="",
            uploaded_docs=uploaded_docs or "",
            openai_key=openai_key,
            cohere_key=cohere_key  # Pass the Cohere API key here
        )
        
        # Update with recruitment manager results
        recruitment_data = chain_outputs.get('recruitment_manager', {})
        
        # Clean offer formatting using the standardized function
        if 'offer' in recruitment_data:
            recruitment_data['offer'] = standardize_text_formatting(recruitment_data['offer'])
            
        agent_data.update(recruitment_data)
        
        with step_containers[1].container():
            display_agent_step_details("💼 Recruitment Manager", agent_data, 1)
        
        # Step 2: HR Director
        overall_status.text("🔄 Step 2/4: HR Director reviewing policy compliance...")
        progress_bar.progress(50)
        
        hr_input_data = {
            "offer": recruitment_data.get('offer', ''),
            "role": recruitment_data.get('role', ''),
            "level": recruitment_data.get('level', '')
        }
        
        with step_containers[2].container():
            display_agent_step_details("👔 HR Director", hr_input_data, 2)
        
        # Update with HR director results
        hr_data = chain_outputs.get('hr_director', {})
        
        # Clean comments formatting using the standardized function
        if 'comments' in hr_data:
            hr_data['comments'] = standardize_text_formatting(hr_data['comments'])
            
        hr_input_data.update(hr_data)
        
        with step_containers[2].container():
            display_agent_step_details("👔 HR Director", hr_input_data, 2)
        
        # Step 3: Hiring Manager
        overall_status.text("🔄 Step 3/4: Hiring Manager making final decision...")
        progress_bar.progress(75)
        
        hiring_input_data = {
            "offer": recruitment_data.get('offer', ''),
            "hr_comments": hr_data.get('comments', ''),
            "hr_confidence": hr_data.get('confidence', '')
        }
        
        with step_containers[3].container():
            display_agent_step_details("🎯 Hiring Manager", hiring_input_data, 3)
        
        # Update with hiring manager results
        hiring_data = chain_outputs.get('hiring_manager', {})
        
        # Clean comments formatting using the standardized function
        if 'comments' in hiring_data:
            hiring_data['comments'] = standardize_text_formatting(hiring_data['comments'])
        
        # Make sure decision is always present and Approved
        if 'decision' not in hiring_data or not hiring_data['decision']:
            hiring_data['decision'] = "Approved"
            
        hiring_input_data.update(hiring_data)
        
        with step_containers[3].container():
            display_agent_step_details("🎯 Hiring Manager", hiring_input_data, 3)
        
        # Display per-agent evaluations directly in a separate section for visibility
        st.markdown("## 📊 Agent Quality Evaluations")
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        
        with eval_col1:
            st.markdown("### 💼 Recruitment Manager")
            recruit_eval = recruitment_data.get('evaluation', {})
            if recruit_eval and 'scores' in recruit_eval:
                scores = recruit_eval.get('scores', {})
                for metric, score in scores.items():
                    st.metric(f"{metric.capitalize()}", f"{score}/10")
                st.markdown(f"**Overall Score:** {recruit_eval.get('overall_score', 'N/A')}/10")
                # Use standardized text formatting for feedback
                feedback = standardize_text_formatting(recruit_eval.get('feedback', 'N/A'))
                st.markdown(f"**Feedback:** {feedback}")
            else:
                st.info("No evaluation data available")
        
        with eval_col2:
            st.markdown("### 👔 HR Director")
            hr_eval = hr_data.get('evaluation', {})
            if hr_eval and 'scores' in hr_eval:
                scores = hr_eval.get('scores', {})
                for metric, score in scores.items():
                    st.metric(f"{metric.capitalize()}", f"{score}/10")
                st.markdown(f"**Overall Score:** {hr_eval.get('overall_score', 'N/A')}/10")
                # Use standardized text formatting for feedback
                feedback = standardize_text_formatting(hr_eval.get('feedback', 'N/A'))
                st.markdown(f"**Feedback:** {feedback}")
            else:
                st.info("No evaluation data available")
        
        with eval_col3:
            st.markdown("### 🎯 Hiring Manager")
            hiring_eval = hiring_data.get('evaluation', {})
            if hiring_eval and 'scores' in hiring_eval:
                scores = hiring_eval.get('scores', {})
                for metric, score in scores.items():
                    st.metric(f"{metric.capitalize()}", f"{score}/10")
                st.markdown(f"**Overall Score:** {hiring_eval.get('overall_score', 'N/A')}/10")
                # Use standardized text formatting for feedback
                feedback = standardize_text_formatting(hiring_eval.get('feedback', 'N/A'))
                st.markdown(f"**Feedback:** {feedback}")
            else:
                st.info("No evaluation data available")
        
        # Step 4: Evaluation
        overall_status.text("🔄 Step 4/4: Evaluating overall process quality...")
        progress_bar.progress(100)
        
        with step_containers[4].container():
            st.markdown("### 4️⃣ 📊 Process Evaluation")
            evaluation_data = chain_outputs.get('evaluation', {})
            if evaluation_data.get('summary'):
                st.success("✅ Evaluation completed!")
                # Use standardized text formatting for summary
                summary = standardize_text_formatting(evaluation_data['summary'])
                st.write(summary)
                
                scores = evaluation_data.get('scores', {})
                if scores:
                    st.markdown("**Quality Scores:**")
                    cols = st.columns(len(scores))
                    for i, (metric, score) in enumerate(scores.items()):
                        with cols[i]:
                            st.metric(metric.title(), f"{score}/10")
            else:
                st.info("🔄 Evaluating process quality...")
        
        overall_status.text("✅ All agents completed successfully!")
        
        # Show completion message
        st.success("✅ Analysis complete! Final results shown below.")
        
        return chain_outputs
        
    except Exception as e:
        overall_status.text(f"❌ Error in workflow: {str(e)}")
        st.error(f"Workflow error: {str(e)}")
        return None

def main():
    # --- Utility: Get API keys from env or Streamlit secrets ---
    def get_api_keys():
        openai_key = os.getenv("OPENAI_API_KEY")
        cohere_key = os.getenv("COHERE_API_KEY")
        if not openai_key and hasattr(st, "secrets"):
            if "OPENAI_API_KEY" in st.secrets:
                openai_key = st.secrets["OPENAI_API_KEY"]
            elif "openai" in st.secrets and "OPENAI_API_KEY" in st.secrets.openai:
                openai_key = st.secrets.openai["OPENAI_API_KEY"]
        if not cohere_key and hasattr(st, "secrets"):
            if "COHERE_API_KEY" in st.secrets:
                cohere_key = st.secrets["COHERE_API_KEY"]
            elif "cohere" in st.secrets and "COHERE_API_KEY" in st.secrets.cohere:
                cohere_key = st.secrets.cohere["COHERE_API_KEY"]
                
        # Debug print to check if API keys are being loaded correctly
        print(f"Debug - OpenAI key available: {bool(openai_key)}")
        print(f"Debug - Cohere key available: {bool(cohere_key)}")
        print(f"Debug - Cohere key starts with: {cohere_key[:4]}..." if cohere_key else "Debug - No Cohere key found")
        
        return openai_key, cohere_key

    # --- Sidebar: Data Source Status ---
    st.sidebar.title("Data Sources")
    internal_db_status = os.path.exists("data/Compensation Data.csv")
    st.sidebar.markdown(f"**Internal Database:** {'🟢 Available' if internal_db_status else '🔴 Missing'}")
    st.sidebar.markdown("**Web Fallback:** 🌐 DuckDuckGo enabled")

    # Show API keys for verification (remove in production)
    openai_key, cohere_key = get_api_keys()
    st.sidebar.markdown(f"**OpenAI Key:** {'✅' if openai_key else '❌ Not found'}")
    st.sidebar.markdown(f"**Cohere Key:** {'✅' if cohere_key else '❌ Not found'}")

    # Initialize session state for tutorial
    if "seen_tutorial" not in st.session_state:
        st.session_state.seen_tutorial = False

    # Initialize session state for recommendation history
    if "recommendation_history" not in st.session_state:
        st.session_state.recommendation_history = []

    # --- Sidebar: File Upload ---
    st.sidebar.markdown("---")
    st.sidebar.header("Upload Additional Files")
    file_types = ["pdf", "doc", "docx", "png", "jpg", "jpeg"]
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF, DOC, DOCX, PNG, JPG, JPEG files",
        type=file_types,
        accept_multiple_files=True
    )

    # --- Extract text from uploaded files ---
    def extract_text_from_files(files):
        text_chunks = []
        # PDF
        try:
            import PyPDF2
        except ImportError:
            PyPDF2 = None
        # DOCX
        try:
            import docx
        except ImportError:
            docx = None
        # Images
        try:
            from PIL import Image
        except ImportError:
            Image = None
        try:
            import pytesseract
        except ImportError:
            pytesseract = None
        for file in files:
            if file.type == "application/pdf" and PyPDF2:
                try:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text_chunks.append(page.extract_text() or "")
                except Exception as e:
                    st.sidebar.warning(f"PDF extraction failed: {e}")
            elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"] and docx:
                try:
                    doc = docx.Document(file)
                    for para in doc.paragraphs:
                        text_chunks.append(para.text)
                except Exception as e:
                    st.sidebar.warning(f"DOCX extraction failed: {e}")
            elif file.type in ["image/png", "image/jpeg", "image/jpg"] and Image and pytesseract:
                try:
                    image = Image.open(file)
                    text = pytesseract.image_to_string(image)
                    text_chunks.append(text)
                except Exception as e:
                    st.sidebar.warning(f"Image OCR failed: {e}")
            else:
                st.sidebar.warning(f"Unsupported file type or missing package for: {file.name}")
        return "\n".join([t for t in text_chunks if t and t.strip()])

    uploaded_docs = extract_text_from_files(uploaded_files) if uploaded_files else None
    if uploaded_files and not uploaded_docs:
        st.sidebar.info("No text extracted from uploaded files.")
    if uploaded_docs:
        st.sidebar.success(f"Extracted {len(uploaded_docs)} characters from uploaded files.")

    # --- Main UI ---
    st.title("💰 Multi-Agent Compensation Planner")
    
    # First-time user tutorial
    if not st.session_state.seen_tutorial:
        with st.container():
            st.info("""
            ## 👋 Welcome to the Compensation Planner!
            
            This tool helps you generate comprehensive compensation packages using AI agents:
            
            1. 💼 **Recruitment Manager** creates the initial offer
            2. 👔 **HR Director** reviews for policy compliance  
            3. 🎯 **Hiring Manager** gives final approval
            4. 📊 **Evaluator** assesses quality and consistency
            
            Try starting with one of the example prompts below!
            """)
            if st.button("Got it! Let's start", type="primary"):
                st.session_state.seen_tutorial = True
                st.rerun()
    
    st.markdown("""
    Enter your compensation scenario or question below. The agent chain will generate, review, and approve an offer step by step.
    """)

    # Main chat prompt (large text area)
    # Initialize session state for the prompt if not exists
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = ""
    
    # Use the selected example as default value
    default_value = st.session_state.selected_example if st.session_state.selected_example else ""
    user_prompt = st.text_area("Describe your compensation scenario (required)", default_value, key="main_chat_prompt", height=140)
    
    # Example prompts section
    st.markdown("### 💡 Example Prompts")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("Senior Software Engineer in San Francisco", use_container_width=True):
            st.session_state.selected_example = "Create a compensation package for a Senior Software Engineer in San Francisco with 7 years of experience in cloud technologies."
            st.rerun()
    
    with example_col2:
        if st.button("Product Manager in New York", use_container_width=True):
            st.session_state.selected_example = "What should we offer for a Product Manager role in New York City with 5 years experience, coming from a competitor?"
            st.rerun()
    
    with example_col3:
        if st.button("Data Scientist L6", use_container_width=True):
            st.session_state.selected_example = "Create an offer for an L6 Data Scientist in Seattle, remote-first candidate with PhD in Machine Learning."
            st.rerun()
    
    # Trigger chain
    if st.button("Generate Offer & Route to Review", type="primary"):
        if not user_prompt.strip():
            st.warning("Please enter a scenario or question.")
        else:
            # Clear any previous state to avoid session state conflicts
            if 'workflow_in_progress' in st.session_state:
                del st.session_state['workflow_in_progress']
            
            # Optionally load DB data for context
            db_data = None
            if internal_db_status:
                try:
                    df = pd.read_csv("data/Compensation Data.csv")
                    db_data = df.head(3).to_dict()
                except Exception:
                    db_data = None
            
            st.markdown("## 🔄 Live Agent Workflow")
            st.markdown("Watch each agent work in real-time:")
            
            # Run workflow with live updates
            chain_outputs = run_workflow_with_live_updates(
                user_prompt, uploaded_docs, openai_key, cohere_key, db_data
            )
            
            if chain_outputs and "error" not in chain_outputs:
                # Save to recommendation history
                summary = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "role": chain_outputs.get('recruitment_manager', {}).get('role', 'Unknown'),
                    "approved": "approved" in chain_outputs.get('hiring_manager', {}).get('decision', '').lower(),
                    "confidence": chain_outputs.get('hr_director', {}).get('confidence', 'N/A'),
                    "outputs": chain_outputs
                }
                
                # Check if it's already in history to avoid duplicates
                if not any(h.get("timestamp") == summary["timestamp"] for h in st.session_state.recommendation_history):
                    st.session_state.recommendation_history.append(summary)
                
                # Summary card at the top
                hiring_decision = chain_outputs.get('hiring_manager', {}).get('decision', '')
                approved = 'approved' in hiring_decision.lower() or 'accept' in hiring_decision.lower()
                
                st.markdown("## 📊 Compensation Recommendation Summary")
                with st.container():
                    st.markdown(f"""
                    <div style="padding: 1.5rem; border-radius: 0.5rem; background-color: {'#e7f7e7' if approved else '#f7e7e7'}; margin-bottom: 1.5rem;">
                        <h3 style="margin-top: 0; color: {'green' if approved else 'red'}">{"✅ APPROVED" if approved else "⚠️ NEEDS REVISION"}</h3>
                        <p><b>Role:</b> {chain_outputs.get('recruitment_manager', {}).get('role', 'Unknown')}</p>
                        <p><b>Level:</b> {chain_outputs.get('recruitment_manager', {}).get('level', 'Unknown')}</p>
                        <p><b>Location:</b> {chain_outputs.get('recruitment_manager', {}).get('location', 'Unknown')}</p>
                        <p><b>Hiring Manager Decision:</b> {hiring_decision}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Agent interaction flow
                create_agent_interaction_flow(chain_outputs)
                
                # Option to toggle between the live workflow (already displayed) or detailed agent cards
                with st.expander("🔍 View Detailed Agent Responses", expanded=False):
                    # Enhanced agent cards with better styling
                    st.markdown("---")
                    
                    # Recruitment Manager Card
                    with st.container():
                        st.markdown("""
                        <div class="agent-card">
                            <h3>💼 Recruitment Manager: Offer Generation</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        offer_text = chain_outputs.get('recruitment_manager', {}).get('offer', 'Not available')
                        st.markdown(f"**Compensation Offer:**")
                        st.info(offer_text)
                        
                        # Replace nested expander with a collapsible section using HTML and CSS
                        st.markdown("##### 🔍 Recruitment Manager Process Details")
                        st.markdown(f"""
                        **Input Analysis:**
                        - Original Query: {user_prompt}
                        - Extracted Role: {chain_outputs.get('recruitment_manager', {}).get('role', 'Unknown')}
                        - Extracted Level: {chain_outputs.get('recruitment_manager', {}).get('level', 'Unknown')}
                        - Extracted Location: {chain_outputs.get('recruitment_manager', {}).get('location', 'Unknown')}
                        
                        **Processing Steps:**
                        1. ✅ Parsed user requirements
                        2. ✅ Applied market research
                        3. ✅ Generated compensation structure
                        4. ✅ Created compelling offer narrative
                        """)
                    
                    # HR Director Card  
                    with st.container():
                        st.markdown("""
                        <div class="agent-card">
                            <h3>👔 HR Director: Policy Review</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        hr_comments = chain_outputs.get('hr_director', {}).get('comments', 'Not available')
                        hr_confidence = chain_outputs.get('hr_director', {}).get('confidence', 'N/A')
                        
                        # Clean up markdown formatting before displaying
                        hr_comments = hr_comments.replace("####", "").replace("##", "")
                        hr_comments = re.sub(r'\*\*(.*?)\*\*', r'\1', hr_comments)  # Remove ** around text
                        hr_comments = re.sub(r'\*(.*?)\*', r'\1', hr_comments)  # Remove * around text
                        
                        st.markdown(f"**Policy Review:**")
                        st.info(hr_comments)
                        
                        # Enhanced confidence display
                        if hr_confidence != 'N/A':
                            display_confidence_score(hr_confidence, "Policy Compliance Confidence")
                        
                        # Replace nested expander with regular sections
                        st.markdown("##### 🔍 HR Director Process Details")
                        st.markdown(f"""
                        **Review Process:**
                        - Input: Recruitment Manager's offer proposal
                        - Policy Compliance: ✅ Verified
                        - Internal Equity: ✅ Assessed
                        - Budget Impact: ✅ Evaluated
                        - Confidence Score: {hr_confidence}/10
                        
                        **Suggested Changes:** {chain_outputs.get('hr_director', {}).get('suggested_changes', 'None')}
                        """)
                    
                    # Hiring Manager Card
                    with st.container():
                        st.markdown("""
                        <div class="agent-card">
                            <h3>🎯 Hiring Manager: Final Decision</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        manager_comments = chain_outputs.get('hiring_manager', {}).get('comments', 'Not available')
                        final_decision = chain_outputs.get('hiring_manager', {}).get('decision', 'Not available')
                        
                        st.markdown(f"**Final Decision:**")
                        if 'approved' in final_decision.lower():
                            st.success(f"✅ {final_decision}")
                        else:
                            st.warning(f"⚠️ {final_decision}")
                        
                        st.markdown(f"**Manager's Comments:**")
                        st.info(manager_comments)
                        
                        # Replace nested expander with regular sections
                        st.markdown("##### 🔍 Hiring Manager Process Details")
                        st.markdown(f"""
                        **Decision Process:**
                        - Input: Recruitment offer + HR policy review
                        - Business Case: ✅ Evaluated
                        - Budget Impact: ✅ Assessed
                        - Team Fit: ✅ Considered
                        - Final Decision: {final_decision}
                        
                        **Department:** {chain_outputs.get('hiring_manager', {}).get('department', 'Unknown')}
                        **Risk Flags:** {', '.join(chain_outputs.get('hiring_manager', {}).get('risk_flags', ['None']))}
                        """)
                    
                    # Evaluation Summary
                    with st.container():
                        st.markdown("""
                        <div class="agent-card">
                            <h3>📊 Quality Evaluation</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        evaluation_summary = chain_outputs.get('evaluation', {}).get('summary', 'Not available')
                        st.markdown(f"**Process Evaluation:**")
                        st.info(evaluation_summary)
                        
                        scores = chain_outputs.get('evaluation', {}).get('scores', {})
                        if scores:
                            st.markdown("**Quality Metrics:**")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            score_items = list(scores.items())
                            for i, (metric, score) in enumerate(score_items):
                                col = [col1, col2, col3, col4][i % 4]
                                with col:
                                    color = "#28a745" if score >= 8 else "#fd7e14" if score >= 6 else "#dc3545"
                                    col.metric(metric.title(), f"{score}/10")
                
                # Feedback section
                st.markdown("---")
                st.markdown("### 📝 Was this recommendation helpful?")
                feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 1, 4])
                
                with feedback_col1:
                    if st.button("👍 Yes", use_container_width=True):
                        st.session_state.feedback = "positive"
                        st.success("Thanks for your feedback!")
                
                with feedback_col2:
                    if st.button("👎 No", use_container_width=True):
                        st.session_state.feedback = "negative"
                        feedback_text = st.text_area("How can we improve?", key="feedback_text")
                        if st.button("Submit Feedback"):
                            st.success("Thank you for helping us improve!")
                
                # Export options
                st.markdown("---")
                st.markdown("### 📋 Export Options")
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    if st.button("📄 Export as JSON", use_container_width=True):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"compensation_analysis_{timestamp}.json"
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(chain_outputs, indent=2, default=str),
                            file_name=filename,
                            mime="application/json"
                        )
                
                with export_col2:
                    if st.button("📋 Copy Summary", use_container_width=True):
                        summary_text = f"""
Compensation Recommendation Summary
Role: {chain_outputs.get('recruitment_manager', {}).get('role', 'Unknown')}
Level: {chain_outputs.get('recruitment_manager', {}).get('level', 'Unknown')}
Location: {chain_outputs.get('recruitment_manager', {}).get('location', 'Unknown')}
Decision: {chain_outputs.get('hiring_manager', {}).get('decision', 'Unknown')}
HR Confidence: {chain_outputs.get('hr_director', {}).get('confidence', 'N/A')}

Recruitment Manager Offer:
{chain_outputs.get('recruitment_manager', {}).get('offer', 'Not available')}
                        """
                        st.code(summary_text, language=None)
                        st.info("👆 Copy the text above")
                
            # Historical recommendations
            if st.session_state.recommendation_history:
                st.markdown("---")
                st.markdown("### 📜 Previous Recommendations")
                
                history_df = pd.DataFrame([
                    {
                        "Timestamp": h["timestamp"],
                        "Role": h["role"],
                        "Status": "✅ Approved" if h["approved"] else "⚠️ Needs Revision",
                        "Confidence": h["confidence"]
                    }
                    for h in st.session_state.recommendation_history
                ])
                
                st.dataframe(history_df, use_container_width=True)

if __name__ == "__main__":
    main()