"""
Enhanced UX Components for Compensation Planner
Workflow visualization, evaluation dashboards, and improved user interface
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

# Try to import streamlit_agraph, but provide a fallback if it's not available
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False
    print("streamlit_agraph not available, using fallback visualization")
    
    # Define dummy classes to avoid errors
    class Node:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class Edge:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    def agraph(*args, **kwargs):
        st.warning("Graph visualization requires streamlit_agraph package. Using text-based fallback.")
        pass


def create_agent_workflow_visualization(workflow_status: Dict[str, Any]) -> Dict[str, Any]:
    """Create interactive workflow visualization showing agent progression"""
    
    # Define nodes for each agent with simpler configuration
    nodes = [
        Node(
            id="recruitment_manager",
            label="Recruitment Manager\n(Draft Offer)",
            size=25,
            color="#FF6B6B" if workflow_status.get("current_step", 0) == 0 else 
                  "#4ECDC4" if workflow_status.get("step_0_complete", False) else "#E0E0E0"
        ),
        Node(
            id="hr_director", 
            label="HR Director\n(Policy Validation)",
            size=25,
            color="#FF6B6B" if workflow_status.get("current_step", 0) == 1 else
                  "#4ECDC4" if workflow_status.get("step_1_complete", False) else "#E0E0E0"
        ),
        Node(
            id="hiring_manager",
            label="Hiring Manager\n(Final Approval)",
            size=25,
            color="#FF6B6B" if workflow_status.get("current_step", 0) == 2 else
                  "#4ECDC4" if workflow_status.get("step_2_complete", False) else "#E0E0E0"
        )
    ]
    
    # Define edges showing workflow progression
    edges = [
        Edge(
            source="recruitment_manager",
            target="hr_director",
            color="#4ECDC4" if workflow_status.get("step_0_complete", False) else "#E0E0E0"
        ),
        Edge(
            source="hr_director",
            target="hiring_manager", 
            color="#4ECDC4" if workflow_status.get("step_1_complete", False) else "#E0E0E0"
        )
    ]
    
    # Simplified configuration to avoid errors
    config = Config(
        width=600,
        height=300,
        directed=True,
        physics=False,  # Disable physics to avoid errors
        hierarchical=False  # Disable hierarchical layout
    )
    
    return {
        "nodes": nodes,
        "edges": edges,
        "config": config
    }


def display_agent_workflow_progress(workflow_result: Dict[str, Any]):
    """Display the agent workflow progress with visual indicators"""
    
    st.markdown("### 🔄 Multi-Agent Workflow Progress")
    
    # Create workflow status
    workflow_status = {
        "current_step": 0,
        "step_0_complete": False,
        "step_1_complete": False, 
        "step_2_complete": False
    }
    
    # Update status based on workflow result
    if workflow_result.get("success", False):
        workflow_status["step_0_complete"] = True
        workflow_status["step_1_complete"] = True
        workflow_status["step_2_complete"] = True
        workflow_status["current_step"] = 3
    
    # Try to create visualization, fallback to simple display if it fails
    try:
        if AGRAPH_AVAILABLE:
            # Create visualization
            workflow_viz = create_agent_workflow_visualization(workflow_status)
            
            # Display the workflow graph
            agraph(
                nodes=workflow_viz["nodes"],
                edges=workflow_viz["edges"], 
                config=workflow_viz["config"]
            )
        else:
            # Fallback to simple text-based progress display
            st.warning("Workflow visualization unavailable - showing text progress instead")
            
            # Simple progress display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if workflow_status["step_0_complete"]:
                    st.success("✅ Recruitment Manager")
                elif workflow_status["current_step"] == 0:
                    st.info("🔄 Recruitment Manager")
                else:
                    st.info("⏳ Recruitment Manager")
            
            with col2:
                if workflow_status["step_1_complete"]:
                    st.success("✅ HR Director")
                elif workflow_status["current_step"] == 1:
                    st.info("🔄 HR Director")
                else:
                    st.info("⏳ HR Director")
            
            with col3:
                if workflow_status["step_2_complete"]:
                    st.success("✅ Hiring Manager")
                elif workflow_status["current_step"] == 2:
                    st.info("🔄 Hiring Manager")
                else:
                    st.info("⏳ Hiring Manager")
    except Exception as e:
        # Fallback to simple text-based progress display
        st.warning(f"Workflow visualization error: {str(e)} - showing text progress instead")
        
        # Simple progress display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if workflow_status["step_0_complete"]:
                st.success("✅ Recruitment Manager")
            elif workflow_status["current_step"] == 0:
                st.info("🔄 Recruitment Manager")
            else:
                st.info("⏳ Recruitment Manager")
        
        with col2:
            if workflow_status["step_1_complete"]:
                st.success("✅ HR Director")
            elif workflow_status["current_step"] == 1:
                st.info("🔄 HR Director")
            else:
                st.info("⏳ HR Director")
        
        with col3:
            if workflow_status["step_2_complete"]:
                st.success("✅ Hiring Manager")
            elif workflow_status["current_step"] == 2:
                st.info("🔄 Hiring Manager")
            else:
                st.info("⏳ Hiring Manager")
    
    # Display workflow steps details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_icon = "✅" if workflow_status["step_0_complete"] else "🔄" if workflow_status["current_step"] == 0 else "⏳"
        st.markdown(f"**{status_icon} Step 1: Draft Offer**")
        st.caption("Recruitment Manager creates initial compensation package")
        
    with col2:
        status_icon = "✅" if workflow_status["step_1_complete"] else "🔄" if workflow_status["current_step"] == 1 else "⏳"
        st.markdown(f"**{status_icon} Step 2: Policy Review**")
        st.caption("HR Director validates compliance and equity")
        
    with col3:
        status_icon = "✅" if workflow_status["step_2_complete"] else "🔄" if workflow_status["current_step"] == 2 else "⏳"
        st.markdown(f"**{status_icon} Step 3: Final Approval**")
        st.caption("Hiring Manager makes final decision")


def create_evaluation_dashboard_ui(evaluation_results: List[Dict[str, Any]], consensus_data: List[Dict[str, Any]] = None):
    """Create comprehensive evaluation dashboard UI"""
    
    if not evaluation_results:
        st.warning("No evaluation data available")
        return
    
    st.markdown("### 📊 Evaluation Dashboard")
    
    # Summary metrics
    latest_eval = evaluation_results[-1]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{latest_eval['overall_score']:.1f}/10",
            delta=f"{latest_eval['overall_score'] - evaluation_results[-2]['overall_score']:.1f}" if len(evaluation_results) > 1 else None
        )
    
    with col2:
        st.metric(
            "Grade",
            latest_eval['grade'],
            delta="Improved" if len(evaluation_results) > 1 and latest_eval['overall_score'] > evaluation_results[-2]['overall_score'] else None
        )
    
    with col3:
        avg_score = np.mean([eval_data['overall_score'] for eval_data in evaluation_results])
        st.metric(
            "Average Score",
            f"{avg_score:.1f}/10"
        )
    
    with col4:
        if consensus_data:
            latest_consensus = consensus_data[-1]
            st.metric(
                "Agent Consensus", 
                f"{latest_consensus.get('consensus_score', 0):.1f}/10"
            )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "🎯 Dimensions", "📋 Detailed Scores", "🤝 Consensus"])
    
    with tab1:
        # Trends over time
        if len(evaluation_results) > 1:
            dates = [eval_data["evaluation_date"] for eval_data in evaluation_results]
            scores = [eval_data["overall_score"] for eval_data in evaluation_results]
            
            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(
                x=dates,
                y=scores,
                mode='lines+markers',
                name='Overall Score',
                line=dict(color='#4ECDC4', width=3),
                marker=dict(size=8)
            ))
            
            fig_trends.update_layout(
                title="Recommendation Quality Trends",
                xaxis_title="Date",
                yaxis_title="Overall Score (0-10)",
                yaxis=dict(range=[0, 10]),
                height=400
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("Need multiple evaluations to show trends")
    
    with tab2:
        # Radar chart for evaluation dimensions
        dimensions = list(latest_eval["dimension_scores"].keys())
        scores = [latest_eval["dimension_scores"][dim]["score"] for dim in dimensions]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=scores,
            theta=[dim.replace('_', ' ').title() for dim in dimensions],
            fill='toself',
            name='Current Recommendation',
            line_color='#FF6B6B'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            title="Evaluation Dimensions Analysis",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Strengths and improvements
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🌟 Strengths:**")
            for strength in latest_eval.get("strengths", []):
                st.markdown(f"• {strength}")
        
        with col2:
            st.markdown("**🎯 Areas for Improvement:**")
            for improvement in latest_eval.get("improvement_areas", []):
                st.markdown(f"• {improvement}")
    
    with tab3:
        # Detailed dimension scores
        st.markdown("**Detailed Dimension Scores:**")
        
        dimension_data = []
        for dim, score_info in latest_eval["dimension_scores"].items():
            dimension_data.append({
                "Dimension": dim.replace('_', ' ').title(),
                "Score": f"{score_info['score']:.1f}",
                "Max Score": f"{score_info['max_score']:.1f}",
                "Weight": f"{score_info['weight']:.0%}",
                "Weighted Score": f"{score_info['weighted_score']:.2f}"
            })
        
        df_dimensions = pd.DataFrame(dimension_data)
        st.dataframe(df_dimensions, use_container_width=True)
        
        # Detailed feedback
        st.markdown("**Detailed Feedback:**")
        for dim, feedback in latest_eval["detailed_feedback"].items():
            with st.expander(f"📝 {dim.replace('_', ' ').title()}"):
                st.write(feedback)
    
    with tab4:
        # Consensus analysis
        if consensus_data:
            latest_consensus = consensus_data[-1]
            
            st.markdown("**Agent Consensus Analysis:**")
            
            # Consensus score visualization
            consensus_score = latest_consensus.get("consensus_score", 0)
            fig_consensus = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = consensus_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Consensus Score"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 8], 'color': "yellow"},
                        {'range': [8, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8
                    }
                }
            ))
            
            fig_consensus.update_layout(height=300)
            st.plotly_chart(fig_consensus, use_container_width=True)
            
            # Agreement and disagreement areas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤝 Agreement Areas:**")
                for agreement in latest_consensus.get("agreement_areas", []):
                    st.markdown(f"• {agreement}")
            
            with col2:
                st.markdown("**⚠️ Disagreement Areas:**")
                for disagreement in latest_consensus.get("disagreement_areas", []):
                    st.markdown(f"• {disagreement}")
        else:
            st.info("Consensus analysis requires multiple agent recommendations")


def create_compensation_comparison_table(recommendations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table for multiple compensation recommendations"""
    
    comparison_data = []
    
    for agent_name, recommendation in recommendations.items():
        # Extract key information from each recommendation
        rec_text = str(recommendation).lower()
        
        # Try to extract salary information
        import re
        salary_matches = re.findall(r'\$?(\d{1,3}(?:,\d{3})*)', rec_text)
        max_salary = 0
        if salary_matches:
            try:
                salaries = [int(match.replace(',', '')) for match in salary_matches if int(match.replace(',', '')) > 50000]
                max_salary = max(salaries) if salaries else 0
            except:
                pass
        
        # Check for components
        components = []
        if "salary" in rec_text or "base" in rec_text:
            components.append("Base Salary")
        if "bonus" in rec_text:
            components.append("Bonus")
        if "equity" in rec_text or "stock" in rec_text:
            components.append("Equity")
        if "benefits" in rec_text:
            components.append("Benefits")
        
        comparison_data.append({
            "Agent": agent_name,
            "Max Salary": f"${max_salary:,}" if max_salary > 0 else "Not specified",
            "Components": ", ".join(components) if components else "Not specified",
            "Confidence": f"{recommendation.get('confidence_score', 5.0):.1f}/10",
            "Length": len(str(recommendation))
        })
    
    return pd.DataFrame(comparison_data)


def display_recommendation_comparison(recommendations: Dict[str, Dict[str, Any]]):
    """Display side-by-side comparison of agent recommendations"""
    
    st.markdown("### 🔄 Agent Recommendation Comparison")
    
    if len(recommendations) < 2:
        st.warning("Need at least 2 recommendations for comparison")
        return
    
    # Create comparison table
    comparison_df = create_compensation_comparison_table(recommendations)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Side-by-side detailed view
    st.markdown("**Detailed Recommendations:**")
    
    cols = st.columns(len(recommendations))
    
    for i, (agent_name, recommendation) in enumerate(recommendations.items()):
        with cols[i]:
            st.markdown(f"**{agent_name}**")
            
            # Display recommendation in an expandable container
            with st.expander("View Full Recommendation", expanded=True):
                if isinstance(recommendation, dict) and "recommendation" in recommendation:
                    st.markdown(recommendation["recommendation"])
                else:
                    st.write(str(recommendation))
            
            # Show confidence and other metrics
            confidence = recommendation.get("confidence_score", 5.0)
            st.progress(confidence / 10)
            st.caption(f"Confidence: {confidence:.1f}/10")


def create_market_data_visualization(market_data: Dict[str, Any]) -> go.Figure:
    """Create visualization for market compensation data"""
    
    if not market_data:
        return go.Figure()
    
    # Extract salary data
    salary_range = market_data.get("base_salary_range", {})
    percentiles = market_data.get("market_percentiles", {})
    
    fig = go.Figure()
    
    # Add salary range bar
    if salary_range:
        fig.add_trace(go.Bar(
            x=["Min Salary", "Max Salary"],
            y=[salary_range.get("min", 0), salary_range.get("max", 0)],
            name="Salary Range",
            marker_color='#4ECDC4'
        ))
    
    # Add percentiles if available
    if percentiles:
        fig.add_trace(go.Scatter(
            x=["P25", "P50", "P75", "P90"],
            y=[percentiles.get("p25", 0), percentiles.get("p50", 0), 
               percentiles.get("p75", 0), percentiles.get("p90", 0)],
            mode='lines+markers',
            name="Market Percentiles",
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Market Compensation Data",
        xaxis_title="Category",
        yaxis_title="Salary ($)",
        height=400
    )
    
    return fig


def display_enhanced_sidebar():
    """Display enhanced sidebar with mode-specific controls and active mode highlight"""
    with st.sidebar:
        st.markdown("### 🎛️ Advanced Controls")
        # Mode selection with highlight
        agent_modes = ["CrewAI Multi-Agent", "Single Agent", "RAG Enhanced", "Evaluation Mode"]
        agent_mode = st.selectbox(
            "Select Mode:",
            options=agent_modes,
            help="Choose how the compensation planning system operates"
        )
        # Visual highlight for active mode
        st.markdown(f"<div style='margin-bottom:10px;'><span style='background-color:#4ECDC4;color:white;padding:4px 12px;border-radius:8px;font-weight:bold;'>Active Mode: {agent_mode}</span></div>", unsafe_allow_html=True)
        # Show only relevant controls for the selected mode
        if agent_mode == "Evaluation Mode":
            st.markdown("**Evaluation Settings:**")
            enable_auto_eval = st.checkbox("Auto-evaluate recommendations", value=True)
            eval_dimensions = st.multiselect(
                "Evaluation Dimensions:",
                options=["Completeness", "Market Competitiveness", "Policy Compliance", "Internal Equity", "Justification Quality", "Clarity", "Risk Assessment"],
                default=["Completeness", "Market Competitiveness", "Policy Compliance"]
            )
        else:
            enable_auto_eval = False
            eval_dimensions = []
        if agent_mode == "RAG Enhanced":
            st.markdown("**RAG Settings:**")
            search_k = st.slider("Search Results", min_value=3, max_value=20, value=5)
            rerank_enabled = st.checkbox("Enable Reranking", value=True)
        else:
            search_k = 5
            rerank_enabled = True
        # Data source controls (show for all modes)
        st.markdown("**Data Sources:**")
        data_sources = st.multiselect(
            "Active Data Sources:",
            options=["Internal Database", "Cohere RAG", "Web Search", "Market APIs"],
            default=["Internal Database", "Cohere RAG"]
        )
        # Quick actions
        st.markdown("**Quick Actions:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset"):
                st.session_state.clear()
                st.experimental_rerun()
        with col2:
            if st.button("📊 Dashboard"):
                st.session_state["show_dashboard"] = True
        # System status
        st.markdown("### 📊 System Status")
        status_data = {
            "CrewAI": "🟢 Active",
            "Cohere RAG": "🟢 Active",
            "Vector DB": "🟢 Connected",
            "Evaluation": "🟢 Ready"
        }
        for system, status in status_data.items():
            st.markdown(f"**{system}**: {status}")
        return {
            "agent_mode": agent_mode,
            "data_sources": data_sources,
            "auto_eval": enable_auto_eval,
            "eval_dimensions": eval_dimensions,
            "search_k": search_k,
            "rerank_enabled": rerank_enabled
        }


def display_workflow_metrics(workflow_data: Dict[str, Any]):
    """Display workflow performance metrics"""
    
    st.markdown("### ⚡ Workflow Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        execution_time = workflow_data.get("execution_time", 0)
        st.metric("Execution Time", f"{execution_time:.1f}s")
    
    with col2:
        tasks_completed = workflow_data.get("tasks_completed", 0)
        st.metric("Tasks Completed", tasks_completed)
    
    with col3:
        agents_involved = len(workflow_data.get("agents_involved", []))
        st.metric("Agents Used", agents_involved)
    
    with col4:
        success_rate = 100 if workflow_data.get("success", False) else 0
        st.metric("Success Rate", f"{success_rate}%")


def create_export_functionality(data: Dict[str, Any], filename_prefix: str = "compensation_analysis"):
    """Create export functionality for analysis results"""
    
    st.markdown("### 📤 Export Results")
    
    export_format = st.selectbox(
        "Export Format:",
        options=["JSON", "CSV", "PDF Report"],
        help="Choose the format for exporting your analysis"
    )
    
    if st.button("Generate Export"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}"
        
        if export_format == "JSON":
            json_str = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{filename}.json",
                mime="application/json"
            )
        
        elif export_format == "CSV":
            if isinstance(data, dict) and "evaluations" in data:
                # Convert evaluation data to CSV
                eval_data = []
                for eval_result in data["evaluations"]:
                    row = {
                        "Date": eval_result["evaluation_date"],
                        "Overall Score": eval_result["overall_score"],
                        "Grade": eval_result["grade"]
                    }
                    # Add dimension scores
                    for dim, score_info in eval_result["dimension_scores"].items():
                        row[f"{dim}_score"] = score_info["score"]
                    eval_data.append(row)
                
                df = pd.DataFrame(eval_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{filename}.csv",
                    mime="text/csv"
                )
        
        elif export_format == "PDF Report":
            st.info("PDF export functionality would be implemented with libraries like reportlab or weasyprint")


def get_enhanced_ui_components():
    """Factory function to get all enhanced UI components"""
    return {
        "workflow_visualization": create_agent_workflow_visualization,
        "evaluation_dashboard": create_evaluation_dashboard_ui,
        "comparison_table": create_compensation_comparison_table,
        "market_visualization": create_market_data_visualization,
        "enhanced_sidebar": display_enhanced_sidebar,
        "workflow_metrics": display_workflow_metrics,
        "export_functionality": create_export_functionality
    }