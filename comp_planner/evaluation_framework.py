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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class EvaluationDimension(str, Enum):
    """Evaluation dimensions for compensation recommendations"""
    COMPLETENESS = "completeness"
    MARKET_COMPETITIVENESS = "market_competitiveness"
    POLICY_COMPLIANCE = "policy_compliance"
    INTERNAL_EQUITY = "internal_equity"
    BUDGET_ALIGNMENT = "budget_alignment"
    CLARITY = "clarity"
    JUSTIFICATION_QUALITY = "justification_quality"
    RISK_ASSESSMENT = "risk_assessment"


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
        
        criteria[EvaluationDimension.COMPLETENESS] = EvaluationCriteria(
            dimension=EvaluationDimension.COMPLETENESS,
            weight=0.15,
            max_score=10.0,
            description="Completeness of compensation package components",
            scoring_guidelines={
                "9-10": "Includes base salary, bonus, equity, benefits, and additional perks",
                "7-8": "Includes base salary, bonus, and equity with some benefits",
                "5-6": "Includes base salary and one other component",
                "3-4": "Only base salary mentioned",
                "0-2": "Incomplete or missing critical components"
            }
        )
        
        criteria[EvaluationDimension.MARKET_COMPETITIVENESS] = EvaluationCriteria(
            dimension=EvaluationDimension.MARKET_COMPETITIVENESS,
            weight=0.20,
            max_score=10.0,
            description="Alignment with current market rates",
            scoring_guidelines={
                "9-10": "Within 5% of market median, highly competitive",
                "7-8": "Within 10% of market median, competitive",
                "5-6": "Within 20% of market median, somewhat competitive",
                "3-4": "20-30% deviation from market rates",
                "0-2": "More than 30% deviation from market rates"
            }
        )
        
        criteria[EvaluationDimension.POLICY_COMPLIANCE] = EvaluationCriteria(
            dimension=EvaluationDimension.POLICY_COMPLIANCE,
            weight=0.15,
            max_score=10.0,
            description="Adherence to company compensation policies",
            scoring_guidelines={
                "9-10": "Fully compliant with all policies",
                "7-8": "Minor policy deviations with valid justification",
                "5-6": "Some policy issues that need approval",
                "3-4": "Multiple policy violations",
                "0-2": "Major policy violations or non-compliance"
            }
        )
        
        criteria[EvaluationDimension.INTERNAL_EQUITY] = EvaluationCriteria(
            dimension=EvaluationDimension.INTERNAL_EQUITY,
            weight=0.15,
            max_score=10.0,
            description="Fairness relative to existing employees",
            scoring_guidelines={
                "9-10": "Maintains excellent internal equity",
                "7-8": "Good internal equity with minor considerations",
                "5-6": "Some potential equity concerns",
                "3-4": "Notable internal equity issues",
                "0-2": "Major internal equity problems"
            }
        )
        
        criteria[EvaluationDimension.JUSTIFICATION_QUALITY] = EvaluationCriteria(
            dimension=EvaluationDimension.JUSTIFICATION_QUALITY,
            weight=0.15,
            max_score=10.0,
            description="Quality of reasoning and market analysis",
            scoring_guidelines={
                "9-10": "Comprehensive analysis with data-backed reasoning",
                "7-8": "Good justification with supporting evidence",
                "5-6": "Basic justification provided",
                "3-4": "Weak or incomplete justification",
                "0-2": "No justification or poor reasoning"
            }
        )
        
        criteria[EvaluationDimension.CLARITY] = EvaluationCriteria(
            dimension=EvaluationDimension.CLARITY,
            weight=0.10,
            max_score=10.0,
            description="Clarity and understandability of recommendation",
            scoring_guidelines={
                "9-10": "Crystal clear, well-structured, easy to understand",
                "7-8": "Clear with minor ambiguities",
                "5-6": "Generally clear but some confusion",
                "3-4": "Somewhat unclear or poorly structured",
                "0-2": "Confusing or very poorly presented"
            }
        )
        
        criteria[EvaluationDimension.RISK_ASSESSMENT] = EvaluationCriteria(
            dimension=EvaluationDimension.RISK_ASSESSMENT,
            weight=0.10,
            max_score=10.0,
            description="Identification and mitigation of risks",
            scoring_guidelines={
                "9-10": "Comprehensive risk analysis with mitigation strategies",
                "7-8": "Good risk identification with some mitigation",
                "5-6": "Basic risk assessment",
                "3-4": "Limited risk consideration",
                "0-2": "No risk assessment provided"
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
        
        if dimension == EvaluationDimension.COMPLETENESS:
            return self._score_completeness(recommendation)
        elif dimension == EvaluationDimension.MARKET_COMPETITIVENESS:
            return self._score_market_competitiveness(recommendation, context)
        elif dimension == EvaluationDimension.POLICY_COMPLIANCE:
            return self._score_policy_compliance(recommendation)
        elif dimension == EvaluationDimension.INTERNAL_EQUITY:
            return self._score_internal_equity(recommendation, context)
        elif dimension == EvaluationDimension.JUSTIFICATION_QUALITY:
            return self._score_justification_quality(recommendation)
        elif dimension == EvaluationDimension.CLARITY:
            return self._score_clarity(recommendation)
        elif dimension == EvaluationDimension.RISK_ASSESSMENT:
            return self._score_risk_assessment(recommendation)
        else:
            return 5.0, "Default scoring for unknown dimension"
    
    def _score_completeness(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score completeness of compensation package"""
        components = []
        
        # Check for key components
        if recommendation.get("base_salary") or "salary" in str(recommendation).lower():
            components.append("base_salary")
        if recommendation.get("bonus") or "bonus" in str(recommendation).lower():
            components.append("bonus")
        if recommendation.get("equity") or "equity" in str(recommendation).lower() or "stock" in str(recommendation).lower():
            components.append("equity")
        if "benefits" in str(recommendation).lower() or "insurance" in str(recommendation).lower():
            components.append("benefits")
        if "vacation" in str(recommendation).lower() or "pto" in str(recommendation).lower():
            components.append("time_off")
        
        component_count = len(components)
        
        if component_count >= 4:
            score = 9.0 + (component_count - 4) * 0.2
            feedback = f"Excellent completeness with {component_count} components identified"
        elif component_count == 3:
            score = 7.5
            feedback = "Good completeness with major components covered"
        elif component_count == 2:
            score = 5.5
            feedback = "Basic completeness, missing some important components"
        elif component_count == 1:
            score = 3.0
            feedback = "Limited completeness, only one component identified"
        else:
            score = 1.0
            feedback = "Poor completeness, no clear components identified"
        
        return min(score, 10.0), feedback
    
    def _score_market_competitiveness(self, recommendation: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        """Score market competitiveness based on available benchmarks"""
        # This would ideally compare against real market data
        # For now, we'll use heuristics based on the recommendation content
        
        rec_text = str(recommendation).lower()
        
        # Look for market references
        market_indicators = [
            "market rate", "competitive", "benchmark", "industry standard",
            "market data", "salary survey", "levels.fyi", "glassdoor"
        ]
        
        market_refs = sum(1 for indicator in market_indicators if indicator in rec_text)
        
        # Look for specific numbers that seem reasonable (heuristic)
        if context and context.get("role") and context.get("level"):
            role = context["role"].lower()
            level = context["level"].lower()
            
            # Basic competitiveness heuristics
            if "senior" in level and "engineer" in role:
                expected_range = (160000, 250000)
            elif "junior" in level and "engineer" in role:
                expected_range = (100000, 140000)
            else:
                expected_range = (120000, 200000)  # Default range
            
            # Try to extract salary numbers from recommendation
            import re
            salary_matches = re.findall(r'\$?(\d{1,3}(?:,\d{3})*)', rec_text)
            if salary_matches:
                try:
                    salaries = [int(match.replace(',', '')) for match in salary_matches if int(match.replace(',', '')) > 50000]
                    if salaries:
                        max_salary = max(salaries)
                        if expected_range[0] <= max_salary <= expected_range[1]:
                            competitiveness_score = 8.5
                        elif expected_range[0] * 0.8 <= max_salary <= expected_range[1] * 1.2:
                            competitiveness_score = 7.0
                        else:
                            competitiveness_score = 5.0
                    else:
                        competitiveness_score = 6.0
                except:
                    competitiveness_score = 6.0
            else:
                competitiveness_score = 5.0
        else:
            competitiveness_score = 6.0
        
        # Adjust based on market references
        final_score = min(competitiveness_score + (market_refs * 0.5), 10.0)
        
        feedback = f"Market competitiveness assessed with {market_refs} market references found"
        
        return final_score, feedback
    
    def _score_policy_compliance(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score policy compliance"""
        rec_text = str(recommendation).lower()
        
        # Look for policy-related content
        policy_indicators = [
            "policy", "compliance", "guidelines", "approval", "budget",
            "authorization", "standard", "exception"
        ]
        
        policy_refs = sum(1 for indicator in policy_indicators if indicator in rec_text)
        
        # Look for potential red flags
        red_flags = ["exceeds", "violation", "non-compliant", "over budget"]
        flag_count = sum(1 for flag in red_flags if flag in rec_text)
        
        base_score = 8.0 if policy_refs > 0 else 6.5
        penalty = flag_count * 1.5
        
        final_score = max(base_score - penalty, 0.0)
        
        feedback = f"Policy compliance score based on {policy_refs} policy references and {flag_count} potential issues"
        
        return final_score, feedback
    
    def _score_internal_equity(self, recommendation: Dict[str, Any], context: Dict[str, Any]) -> Tuple[float, str]:
        """Score internal equity considerations"""
        rec_text = str(recommendation).lower()
        
        equity_indicators = [
            "internal equity", "fair", "consistent", "peer", "team",
            "existing employees", "pay equity", "compensation band"
        ]
        
        equity_refs = sum(1 for indicator in equity_indicators if indicator in rec_text)
        
        if equity_refs >= 2:
            score = 8.5
            feedback = "Good consideration of internal equity"
        elif equity_refs == 1:
            score = 6.5
            feedback = "Some internal equity consideration"
        else:
            score = 5.0
            feedback = "Limited internal equity analysis"
        
        return score, feedback
    
    def _score_justification_quality(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score the quality of justification provided"""
        rec_text = str(recommendation).lower()
        
        # Look for quality justification indicators
        quality_indicators = [
            "because", "due to", "based on", "analysis", "data",
            "research", "benchmark", "market", "experience", "skills"
        ]
        
        justification_score = sum(1 for indicator in quality_indicators if indicator in rec_text)
        
        # Check for data/numbers
        import re
        numbers = re.findall(r'\d+', rec_text)
        has_quantitative = len(numbers) > 2
        
        base_score = min(justification_score * 1.2, 8.0)
        if has_quantitative:
            base_score += 1.0
        
        final_score = min(base_score, 10.0)
        
        feedback = f"Justification quality based on {justification_score} reasoning indicators and quantitative data: {has_quantitative}"
        
        return final_score, feedback
    
    def _score_clarity(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score clarity and structure of recommendation"""
        rec_text = str(recommendation)
        
        # Basic readability metrics
        sentence_count = rec_text.count('.') + rec_text.count('!') + rec_text.count('?')
        word_count = len(rec_text.split())
        
        # Structure indicators
        structure_indicators = ['•', '-', '1.', '2.', '\n', 'Recommendation:', 'Summary:']
        structure_score = sum(1 for indicator in structure_indicators if indicator in rec_text)
        
        # Calculate readability (simplified)
        if word_count > 0 and sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            readability_score = max(10 - (avg_sentence_length - 15) * 0.2, 0)
        else:
            readability_score = 5.0
        
        structure_bonus = min(structure_score * 0.5, 2.0)
        final_score = min(readability_score + structure_bonus, 10.0)
        
        feedback = f"Clarity score based on structure ({structure_score} indicators) and readability"
        
        return final_score, feedback
    
    def _score_risk_assessment(self, recommendation: Dict[str, Any]) -> Tuple[float, str]:
        """Score risk assessment quality"""
        rec_text = str(recommendation).lower()
        
        risk_indicators = [
            "risk", "challenge", "concern", "mitigation", "volatility",
            "budget constraint", "market change", "retention", "competition"
        ]
        
        risk_refs = sum(1 for indicator in risk_indicators if indicator in rec_text)
        
        if risk_refs >= 3:
            score = 9.0
            feedback = "Comprehensive risk assessment"
        elif risk_refs == 2:
            score = 7.0
            feedback = "Good risk consideration"
        elif risk_refs == 1:
            score = 5.0
            feedback = "Basic risk awareness"
        else:
            score = 3.0
            feedback = "Limited risk assessment"
        
        return score, feedback
    
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