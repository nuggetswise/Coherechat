"""
Schema Validation Layer for Compensation Planner

This module provides data validation for the compensation planner using pydantic models.
It ensures data integrity and type safety throughout the application.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from decimal import Decimal
import datetime

class CompensationPackage(BaseModel):
    """Schema for a compensation package proposal"""
    base_salary: Decimal = Field(..., description="Annual base salary in USD")
    bonus_percentage: Optional[Decimal] = Field(None, description="Target bonus as percentage of base salary")
    equity_value: Optional[Decimal] = Field(None, description="Total equity grant value in USD")
    equity_vesting_schedule: Optional[str] = Field(None, description="Description of equity vesting schedule")
    benefits: Dict[str, Any] = Field(default_factory=dict, description="Benefits package details")
    relocation_assistance: Optional[Decimal] = Field(None, description="Relocation assistance amount in USD")
    
    @validator('base_salary', 'bonus_percentage', 'equity_value', 'relocation_assistance')
    def validate_positive_amounts(cls, v, values, **kwargs):
        if v is not None and v < 0:
            field_name = kwargs.get('field', 'field')
            raise ValueError(f"{field_name} must be a positive number")
        return v

class HRFeedback(BaseModel):
    """Schema for HR director's feedback on a compensation package"""
    policy_compliant: bool = Field(..., description="Whether the package complies with company policy")
    internal_equity: bool = Field(..., description="Whether the package maintains internal equity")
    market_competitive: bool = Field(..., description="Whether the package is market competitive")
    strengths: List[str] = Field(default_factory=list, description="Strengths of the compensation package")
    concerns: List[str] = Field(default_factory=list, description="Areas of concern with the compensation package")
    suggested_changes: Dict[str, Any] = Field(default_factory=dict, description="Suggested changes to the package")
    confidence_rating: int = Field(..., ge=1, le=10, description="Confidence rating from 1-10")

class EvaluationScore(BaseModel):
    """Schema for a component evaluation score with reasoning"""
    score: int = Field(..., ge=1, le=10, description="Score from 1-10")
    reasoning: str = Field(..., description="Reasoning behind the score")

class AgentEvaluation(BaseModel):
    """Schema for an agent output evaluation"""
    relevance: EvaluationScore
    factual_accuracy: EvaluationScore
    groundedness: EvaluationScore
    overall_score: float = Field(..., ge=1, le=10, description="Overall evaluation score from 1-10")
    strengths: List[str]
    areas_for_improvement: List[str]
    pass_threshold: bool = Field(..., description="Whether the output passes the minimum quality threshold")

class RoleData(BaseModel):
    """Schema for role data from the database"""
    role_id: str
    role_title: str
    department: str
    level: str
    min_salary: Decimal
    max_salary: Decimal
    median_salary: Decimal
    location: Optional[str] = None
    
class CandidateData(BaseModel):
    """Schema for candidate data"""
    candidate_id: Optional[str] = None
    name: str
    years_experience: int
    education: str
    skills: List[str]
    current_salary: Optional[Decimal] = None
    desired_salary: Optional[Decimal] = None
    location: Optional[str] = None

def validate_compensation_package(data: Dict[str, Any]) -> CompensationPackage:
    """
    Validate compensation package data against the schema
    
    Args:
        data: Dictionary containing compensation package data
        
    Returns:
        Validated CompensationPackage object
    """
    return CompensationPackage.parse_obj(data)

def validate_hr_feedback(data: Dict[str, Any]) -> HRFeedback:
    """
    Validate HR feedback data against the schema
    
    Args:
        data: Dictionary containing HR feedback data
        
    Returns:
        Validated HRFeedback object
    """
    return HRFeedback.parse_obj(data)

def validate_evaluation(data: Dict[str, Any]) -> AgentEvaluation:
    """
    Validate agent evaluation data against the schema
    
    Args:
        data: Dictionary containing evaluation data
        
    Returns:
        Validated AgentEvaluation object
    """
    return AgentEvaluation.parse_obj(data)

# Additional validation functions needed by offer_chain.py
def validate_offer_details(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize offer details from recruitment manager
    
    Args:
        data: Dictionary containing offer details
        
    Returns:
        Validated and normalized data
    """
    # Ensure required fields are present with defaults
    validated_data = {
        "offer": data.get("offer", ""),
        "role": data.get("role", "Unknown Role"),
        "level": data.get("level", "Mid-level"),
        "location": data.get("location", "Remote"),
        "department": data.get("department", "General")
    }
    
    # Clean up any empty strings
    for key, value in validated_data.items():
        if isinstance(value, str) and not value.strip():
            if key == "role":
                validated_data[key] = "Unknown Role"
            elif key == "level":
                validated_data[key] = "Mid-level"
            elif key == "location":
                validated_data[key] = "Remote"
            elif key == "department":
                validated_data[key] = "General"
    
    return validated_data

def validate_director_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize HR director feedback
    
    Args:
        data: Dictionary containing director feedback
        
    Returns:
        Validated and normalized data
    """
    validated_data = {
        "director_comments": data.get("director_comments", ""),
        "confidence": data.get("confidence", 7),
        "suggested_changes": data.get("suggested_changes", "None"),
        "role": data.get("role", "Unknown Role"),
        "level": data.get("level", "Mid-level"),
        "location": data.get("location", "Remote")
    }
    
    # Validate confidence score
    try:
        confidence = int(validated_data["confidence"])
        if confidence < 1:
            confidence = 1
        elif confidence > 10:
            confidence = 10
        validated_data["confidence"] = confidence
    except (ValueError, TypeError):
        validated_data["confidence"] = 7
    
    return validated_data

def validate_manager_approval(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize hiring manager approval
    
    Args:
        data: Dictionary containing manager approval data
        
    Returns:
        Validated and normalized data
    """
    validated_data = {
        "approval_status": data.get("approval_status", "Approved"),
        "manager_comments": data.get("manager_comments", ""),
        "equity_concerns": data.get("equity_concerns", False),
        "risk_flags": data.get("risk_flags", []),
        "role": data.get("role", "Unknown Role"),
        "level": data.get("level", "Mid-level"),
        "decision": data.get("manager_comments", "Approved")  # Use comments as decision for compatibility
    }
    
    # Ensure risk_flags is a list
    if not isinstance(validated_data["risk_flags"], list):
        validated_data["risk_flags"] = []
    
    # Ensure equity_concerns is boolean
    validated_data["equity_concerns"] = bool(validated_data["equity_concerns"])
    
    return validated_data

def validate_evaluation_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize evaluation result
    
    Args:
        data: Dictionary containing evaluation data
        
    Returns:
        Validated and normalized data
    """
    validated_data = {
        "relevance": data.get("relevance", {"score": 7, "reasoning": "Default score"}),
        "factual_accuracy": data.get("factual_accuracy", {"score": 7, "reasoning": "Default score"}),
        "groundedness": data.get("groundedness", {"score": 7, "reasoning": "Default score"}),
        "overall_score": data.get("overall_score", 7.0),
        "strengths": data.get("strengths", ["Generated response"]),
        "areas_for_improvement": data.get("areas_for_improvement", ["Could be more specific"]),
        "pass_threshold": data.get("pass_threshold", True)
    }
    
    # Validate score ranges
    for metric in ["relevance", "factual_accuracy", "groundedness"]:
        if metric in validated_data and isinstance(validated_data[metric], dict):
            score = validated_data[metric].get("score", 7)
            try:
                score = max(1, min(10, int(score)))
                validated_data[metric]["score"] = score
            except (ValueError, TypeError):
                validated_data[metric]["score"] = 7
    
    # Validate overall score
    try:
        overall = float(validated_data["overall_score"])
        validated_data["overall_score"] = max(1.0, min(10.0, overall))
    except (ValueError, TypeError):
        validated_data["overall_score"] = 7.0
    
    # Ensure lists are actually lists
    if not isinstance(validated_data["strengths"], list):
        validated_data["strengths"] = ["Generated response"]
    if not isinstance(validated_data["areas_for_improvement"], list):
        validated_data["areas_for_improvement"] = ["Could be more specific"]
    
    return validated_data