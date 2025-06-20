"""
Comprehensive Evaluation Framework for Compensation Planning
Implements systematic scoring, benchmarking, and agent consensus measurement
"""
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import json
from dataclasses import dataclass
from enum import Enum

# Optional plotly imports - gracefully handle missing plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Create dummy classes to prevent errors
    class DummyPlotly:
        def __getattr__(self, name):
            return lambda *args, **kwargs: {"error": "Plotly not available"}
    
    go = DummyPlotly()
    px = DummyPlotly()
    make_subplots = lambda *args, **kwargs: {"error": "Plotly not available"}


class EvaluationDimension(str, Enum):
    """Evaluation dimensions for compensation recommendations"""
    CONTEXT_RELEVANCE = "context_relevance"
    FAITHFULNESS = "faithfulness"
    CONTEXT_SUPPORT_COVERAGE = "context_support_coverage"
    QUESTION_ANSWERABILITY = "question_answerability"


@dataclass
class EvaluationCriteria:
    """Criteria for each evaluation dimension"""
    dimension: EvaluationDimension
    weight: float
    max_score: float
    description: str
    scoring_guidelines: Dict[str, str]


class CompensationEvaluator:
    """Systematic evaluator for compensation recommendations"""
    
    def __init__(self):
        self.criteria = self._setup_evaluation_criteria()
        self.evaluation_history = []
        
    def _setup_evaluation_criteria(self) -> Dict[EvaluationDimension, EvaluationCriteria]:
        """Setup evaluation criteria and scoring guidelines"""
        criteria = {}
        
        criteria[EvaluationDimension.CONTEXT_RELEVANCE] = EvaluationCriteria(
            dimension=EvaluationDimension.CONTEXT_RELEVANCE,
            weight=0.25,
            max_score=10.0,
            description="Relevance of the recommendation to the given context",
            scoring_guidelines={
                "9-10": "Highly relevant, directly addresses context",
                "7-8": "Mostly relevant, few minor context gaps",
                "5-6": "Some relevance, but notable context gaps",
                "3-4": "Limited relevance, does not fully address context",
                "0-2": "Irrelevant or off-topic"
            }
        )
        
        criteria[EvaluationDimension.FAITHFULNESS] = EvaluationCriteria(
            dimension=EvaluationDimension.FAITHFULNESS,
            weight=0.25,
            max_score=10.0,
            description="Faithfulness to the original intent and details of the request",
            scoring_guidelines={
                "9-10": "Fully faithful to the request, no deviations",
                "7-8": "Mostly faithful, minor deviations",
                "5-6": "Somewhat faithful, but notable deviations",
                "3-4": "Limited faithfulness, significant deviations",
                "0-2": "Not faithful to the request"
            }
        )
        
        criteria[EvaluationDimension.CONTEXT_SUPPORT_COVERAGE] = EvaluationCriteria(
            dimension=EvaluationDimension.CONTEXT_SUPPORT_COVERAGE,
            weight=0.25,
            max_score=10.0,
            description="Coverage of supporting details and context",
            scoring_guidelines={
                "9-10": "Comprehensive coverage of all necessary details",
                "7-8": "Good coverage, few minor details missing",
                "5-6": "Some coverage, but several important details missing",
                "3-4": "Limited coverage, many details missing",
                "0-2": "Very poor coverage, lacks essential details"
            }
        )
        
        criteria[EvaluationDimension.QUESTION_ANSWERABILITY] = EvaluationCriteria(
            dimension=EvaluationDimension.QUESTION_ANSWERABILITY,
            weight=0.25,
            max_score=10.0,
            description="Ability to answer potential questions and objections",
            scoring_guidelines={
                "9-10": "Anticipates and answers all potential questions",
                "7-8": "Covers most questions, few minor gaps",
                "5-6": "Some questions answered, but notable gaps",
                "3-4": "Limited question handling, many gaps",
                "0-2": "Does not address potential questions"
            }
        )
        
        return criteria
    
    def evaluate_recommendation(self, recommendation: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate a compensation recommendation across all dimensions"""
        
        scores = {}
        detailed_feedback = {}
        
        for dimension, criteria in self.criteria.items():
            score, feedback = self._score_dimension(dimension, recommendation, context)
            scores[dimension.value] = {
                "score": score,
                "max_score": criteria.max_score,
                "weight": criteria.weight,
                "weighted_score": score * criteria.weight
            }
            detailed_feedback[dimension.value] = feedback
        
        # Calculate overall score
        total_weighted_score = sum(scores[dim]["weighted_score"] for dim in scores)
        max_possible_score = sum(criteria.weight * criteria.max_score for criteria in self.criteria.values())
        overall_score = (total_weighted_score / max_possible_score) * 10
        
        evaluation_result = {
            "overall_score": round(overall_score, 2),
            "max_score": 10.0,
            "dimension_scores": scores,
            "detailed_feedback": detailed_feedback,
            "evaluation_date": datetime.now().isoformat(),
            "recommendation_id": recommendation.get("id", "unknown"),
            "strengths": self._identify_strengths(scores),
            "improvement_areas": self._identify_improvements(scores),
            "grade": self._assign_grade(overall_score)
        }
        
        # Store in history
        self.evaluation_history.append(evaluation_result)
        
        return evaluation_result
    
    def _score_dimension(self, dimension: EvaluationDimension, recommendation: Dict[str, Any], context: Dict[str, Any] = None) -> Tuple[float, str]:
        """Score a specific dimension of the recommendation"""
        
        if dimension == EvaluationDimension.CONTEXT_RELEVANCE:
            return self._score_context_relevance(recommendation, context)
        elif dimension == EvaluationDimension.FAITHFULNESS:
            return self._score_faithfulness(recommendation)
        elif dimension == EvaluationDimension.CONTEXT_SUPPORT_COVERAGE:
            return self._score_context_support_coverage(recommendation, context)
        elif dimension == EvaluationDimension.QUESTION_ANSWERABILITY:
            return self._score_question_answerability(recommendation)
        else:
            return 5.0, "Default scoring for unknown dimension"
    
    def _score_context_relevance(self, recommendation: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        """Score relevance of the recommendation to the given context"""
        rec_text = str(recommendation).lower()
        
        # Check for presence of context in the recommendation
        if context and context.get("key_points"):
            key_points = [point.lower() for point in context["key_points"]]
            relevance_score = sum(1 for point in key_points if point in rec_text)
            
            if relevance_score == len(key_points):
                return 10.0, "Fully addresses all key points of the context"
            elif relevance_score > 0:
                return 5.0 + (relevance_score / len(key_points)) * 5.0, f"Addresses {relevance_score} out of {len(key_points)} key points"
        
        return 0.0, "Does not address the given context"
    
    def _score_faithfulness(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score faithfulness to the original intent and details of the request"""
        rec_text = str(recommendation).lower()
        
        # Look for deviations from the request
        deviations = ["not mentioned", "excluded", "ignored", "overlooked"]
        deviation_count = sum(1 for dev in deviations if dev in rec_text)
        
        if deviation_count == 0:
            return 10.0, "Fully faithful to the request"
        elif deviation_count == 1:
            return 7.0, "Minor deviations from the request"
        elif deviation_count <= 3:
            return 4.0, "Notable deviations from the request"
        else:
            return 0.0, "Significant deviations, not faithful to the request"
    
    def _score_context_support_coverage(self, recommendation: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        """Score coverage of supporting details and context"""
        rec_text = str(recommendation).lower()
        
        # Check for supporting details
        if context and context.get("supporting_details"):
            details = [detail.lower() for detail in context["supporting_details"]]
            coverage_score = sum(1 for detail in details if detail in rec_text)
            
            if coverage_score == len(details):
                return 10.0, "Covers all supporting details"
            elif coverage_score > 0:
                return 5.0 + (coverage_score / len(details)) * 5.0, f"Covers {coverage_score} out of {len(details)} details"
        
        return 0.0, "Does not cover the necessary supporting details"
    
    def _score_question_answerability(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score ability to answer potential questions and objections"""
        rec_text = str(recommendation).lower()
        
        # Look for question handling indicators
        question_indicators = [
            "if", "when", "where", "how", "what if", "suppose that",
            "considering", "assuming", "in the event that"
        ]
        
        question_score = sum(1 for indicator in question_indicators if indicator in rec_text)
        
        # Check for completeness of answers
        import re
        answer_matches = re.findall(r'\$?(\d{1,3}(?:,\d{3})*)', rec_text)
        has_complete_answers = len(answer_matches) > 2
        
        base_score = min(question_score * 1.2, 8.0)
        if has_complete_answers:
            base_score += 1.0
        
        final_score = min(base_score, 10.0)
        
        feedback = f"Question answerability based on {question_score} indicators and completeness of answers: {has_complete_answers}"
        
        return final_score, feedback
    
    def _identify_strengths(self, scores: Dict[str, Any]) -> List[str]:
        """Identify strengths based on dimension scores"""
        strengths = []
        
        for dimension, score_info in scores.items():
            if score_info["score"] >= 8.0:
                strengths.append(f"Excellent {dimension.replace('_', ' ')}")
        
        return strengths
    
    def _identify_improvements(self, scores: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        for dimension, score_info in scores.items():
            if score_info["score"] < 6.0:
                improvements.append(f"Improve {dimension.replace('_', ' ')}")
        
        return improvements
    
    def _assign_grade(self, overall_score: float) -> str:
        """Assign letter grade based on overall score"""
        if overall_score >= 9.0:
            return "A+"
        elif overall_score >= 8.5:
            return "A"
        elif overall_score >= 8.0:
            return "A-"
        elif overall_score >= 7.5:
            return "B+"
        elif overall_score >= 7.0:
            return "B"
        elif overall_score >= 6.5:
            return "B-"
        elif overall_score >= 6.0:
            return "C+"
        elif overall_score >= 5.5:
            return "C"
        elif overall_score >= 5.0:
            return "C-"
        else:
            return "D"


class AgentConsensusAnalyzer:
    """Analyzer for measuring agreement/disagreement between agents"""
    
    def __init__(self):
        self.consensus_history = []
    
    def analyze_agent_consensus(self, agent_recommendations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consensus between multiple agent recommendations"""
        
        if len(agent_recommendations) < 2:
            return {"error": "Need at least 2 agent recommendations for consensus analysis"}
        
        # Extract key metrics from each recommendation
        agent_metrics = {}
        for agent_name, recommendation in agent_recommendations.items():
            metrics = self._extract_metrics(recommendation)
            agent_metrics[agent_name] = metrics
        
        # Calculate consensus scores
        consensus_analysis = {
            "salary_consensus": self._calculate_salary_consensus(agent_metrics),
            "component_consensus": self._calculate_component_consensus(agent_metrics),
            "overall_consensus": 0.0,
            "disagreement_areas": [],
            "agreement_areas": [],
            "recommendation_variance": {},
            "consensus_score": 0.0
        }
        
        # Calculate overall consensus
        consensus_scores = [
            consensus_analysis["salary_consensus"]["consensus_score"],
            consensus_analysis["component_consensus"]["consensus_score"]
        ]
        consensus_analysis["overall_consensus"] = np.mean(consensus_scores)
        consensus_analysis["consensus_score"] = consensus_analysis["overall_consensus"]
        
        # Identify agreement and disagreement areas
        consensus_analysis["agreement_areas"] = self._identify_agreements(agent_metrics)
        consensus_analysis["disagreement_areas"] = self._identify_disagreements(agent_metrics)
        
        # Store in history
        consensus_result = {
            **consensus_analysis,
            "analysis_date": datetime.now().isoformat(),
            "agents_analyzed": list(agent_recommendations.keys()),
            "agent_count": len(agent_recommendations)
        }
        
        self.consensus_history.append(consensus_result)
        
        return consensus_result
    
    def _extract_metrics(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from a recommendation"""
        rec_text = str(recommendation).lower()
        
        # Extract salary information
        import re
        salary_matches = re.findall(r'\$?(\d{1,3}(?:,\d{3})*)', rec_text)
        salaries = []
        
        if salary_matches:
            try:
                salaries = [int(match.replace(',', '')) for match in salary_matches if int(match.replace(',', '')) > 50000]
            except:
                pass
        
        # Extract components mentioned
        components = []
        if "salary" in rec_text or "base" in rec_text:
            components.append("base_salary")
        if "bonus" in rec_text:
            components.append("bonus")
        if "equity" in rec_text or "stock" in rec_text:
            components.append("equity")
        if "benefits" in rec_text:
            components.append("benefits")
        
        return {
            "salaries": salaries,
            "max_salary": max(salaries) if salaries else 0,
            "components": components,
            "component_count": len(components),
            "recommendation_length": len(str(recommendation)),
            "confidence": recommendation.get("confidence_score", 5.0)
        }
    
    def _calculate_salary_consensus(self, agent_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus on salary recommendations"""
        salaries = []
        
        for agent, metrics in agent_metrics.items():
            if metrics["max_salary"] > 0:
                salaries.append(metrics["max_salary"])
        
        if len(salaries) < 2:
            return {
                "consensus_score": 0.0,
                "salary_range": None,
                "variance": 0.0,
                "agreement_level": "insufficient_data"
            }
        
        mean_salary = np.mean(salaries)
        std_salary = np.std(salaries)
        variance_pct = (std_salary / mean_salary) * 100 if mean_salary > 0 else 100
        
        # Calculate consensus score (higher is better agreement)
        if variance_pct < 5:
            consensus_score = 10.0
            agreement_level = "excellent"
        elif variance_pct < 10:
            consensus_score = 8.5
            agreement_level = "good"
        elif variance_pct < 20:
            consensus_score = 6.5
            agreement_level = "moderate"
        elif variance_pct < 30:
            consensus_score = 4.0
            agreement_level = "poor"
        else:
            consensus_score = 2.0
            agreement_level = "very_poor"
        
        return {
            "consensus_score": consensus_score,
            "salary_range": {"min": min(salaries), "max": max(salaries), "mean": mean_salary},
            "variance_percent": variance_pct,
            "agreement_level": agreement_level,
            "agent_salaries": {agent: metrics["max_salary"] for agent, metrics in agent_metrics.items()}
        }
    
    def _calculate_component_consensus(self, agent_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus on compensation components"""
        all_components = set()
        agent_components = {}
        
        for agent, metrics in agent_metrics.items():
            components = set(metrics["components"])
            all_components.update(components)
            agent_components[agent] = components
        
        if not all_components:
            return {
                "consensus_score": 0.0,
                "common_components": [],
                "unique_components": {},
                "agreement_level": "no_components"
            }
        
        # Calculate component overlap
        common_components = set.intersection(*agent_components.values()) if agent_components else set()
        consensus_score = (len(common_components) / len(all_components)) * 10 if all_components else 0
        
        # Identify unique components per agent
        unique_components = {}
        for agent, components in agent_components.items():
            unique = components - common_components
            if unique:
                unique_components[agent] = list(unique)
        
        agreement_level = "excellent" if consensus_score >= 8 else "good" if consensus_score >= 6 else "moderate" if consensus_score >= 4 else "poor"
        
        return {
            "consensus_score": consensus_score,
            "common_components": list(common_components),
            "unique_components": unique_components,
            "agreement_level": agreement_level,
            "total_components": len(all_components)
        }
    
    def _identify_agreements(self, agent_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify areas where agents agree"""
        agreements = []
        
        # Check component agreement
        all_components = [set(metrics["components"]) for metrics in agent_metrics.values()]
        if all_components:
            common = set.intersection(*all_components)
            if common:
                agreements.append(f"All agents agree on including: {', '.join(common)}")
        
        # Check confidence alignment
        confidences = [metrics["confidence"] for metrics in agent_metrics.values()]
        if confidences and max(confidences) - min(confidences) < 2:
            agreements.append("Agents have similar confidence levels")
        
        return agreements
    
    def _identify_disagreements(self, agent_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify areas where agents disagree"""
        disagreements = []
        
        # Check salary disagreement
        salaries = [metrics["max_salary"] for metrics in agent_metrics.values() if metrics["max_salary"] > 0]
        if len(salaries) >= 2:
            variance_pct = (np.std(salaries) / np.mean(salaries)) * 100
            if variance_pct > 15:
                disagreements.append(f"Significant salary variance: {variance_pct:.1f}%")
        
        # Check component disagreement
        all_components = [set(metrics["components"]) for metrics in agent_metrics.values()]
        if all_components:
            unique_per_agent = 0
            for i, components in enumerate(all_components):
                others = set.union(*[comp for j, comp in enumerate(all_components) if j != i])
                unique = components - others
                unique_per_agent += len(unique)
            
            if unique_per_agent > 0:
                disagreements.append(f"Agents disagree on {unique_per_agent} compensation components")
        
        return disagreements


def create_evaluation_dashboard(evaluations: List[Dict[str, Any]], consensus_analyses: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create comprehensive evaluation dashboard with visualizations"""
    
    if not evaluations:
        return {"error": "No evaluations to display"}
    
    # Check if plotly is available for visualizations
    if not PLOTLY_AVAILABLE:
        # Return dashboard with text-based analysis when plotly unavailable
        dates = [eval_data["evaluation_date"] for eval_data in evaluations]
        scores = [eval_data["overall_score"] for eval_data in evaluations]
        
        dashboard = {
            "charts_available": False,
            "plotly_error": "Plotly not installed - visualizations unavailable",
            "summary_stats": {
                "total_evaluations": len(evaluations),
                "average_score": np.mean(scores),
                "latest_score": scores[-1] if scores else 0,
                "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[-2] else "declining" if len(scores) > 1 else "stable"
            }
        }
        
        # Add consensus analysis if available
        if consensus_analyses:
            latest_consensus = consensus_analyses[-1]
            dashboard["consensus_summary"] = {
                "latest_consensus_score": latest_consensus.get("consensus_score", 0),
                "agents_count": latest_consensus.get("agent_count", 0),
                "agreement_areas": latest_consensus.get("agreement_areas", []),
                "disagreement_areas": latest_consensus.get("disagreement_areas", [])
            }
        
        return dashboard
    
    # Create evaluation trends chart
    fig_trends = go.Figure()
    
    dates = [eval_data["evaluation_date"] for eval_data in evaluations]
    scores = [eval_data["overall_score"] for eval_data in evaluations]
    
    fig_trends.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='blue', width=3)
    ))
    
    fig_trends.update_layout(
        title="Compensation Recommendation Quality Trends",
        xaxis_title="Date",
        yaxis_title="Overall Score (0-10)",
        yaxis=dict(range=[0, 10])
    )
    
    # Create dimension scores radar chart for latest evaluation
    latest_eval = evaluations[-1]
    dimensions = list(latest_eval["dimension_scores"].keys())
    scores = [latest_eval["dimension_scores"][dim]["score"] for dim in dimensions]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=scores,
        theta=dimensions,
        fill='toself',
        name='Current Recommendation'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        title="Evaluation Dimensions - Latest Recommendation"
    )
    
    # Create grade distribution
    grades = [eval_data["grade"] for eval_data in evaluations]
    grade_counts = pd.Series(grades).value_counts()
    
    fig_grades = go.Figure(data=[
        go.Bar(x=grade_counts.index, y=grade_counts.values)
    ])
    fig_grades.update_layout(
        title="Grade Distribution",
        xaxis_title="Grade",
        yaxis_title="Count"
    )
    
    dashboard = {
        "charts_available": True,
        "trends_chart": fig_trends,
        "radar_chart": fig_radar,
        "grades_chart": fig_grades,
        "summary_stats": {
            "total_evaluations": len(evaluations),
            "average_score": np.mean(scores),
            "latest_score": scores[-1] if scores else 0,
            "score_trend": "improving" if len(scores) > 1 and scores[-1] > scores[-2] else "declining" if len(scores) > 1 else "stable"
        }
    }
    
    # Add consensus analysis if available
    if consensus_analyses:
        latest_consensus = consensus_analyses[-1]
        dashboard["consensus_summary"] = {
            "latest_consensus_score": latest_consensus.get("consensus_score", 0),
            "agents_count": latest_consensus.get("agent_count", 0),
            "agreement_areas": latest_consensus.get("agreement_areas", []),
            "disagreement_areas": latest_consensus.get("disagreement_areas", [])
        }
    
    return dashboard


def get_compensation_evaluator() -> CompensationEvaluator:
    """Factory function to create compensation evaluator"""
    return CompensationEvaluator()


def get_agent_consensus_analyzer() -> AgentConsensusAnalyzer:
    """Factory function to create agent consensus analyzer"""
    return AgentConsensusAnalyzer()