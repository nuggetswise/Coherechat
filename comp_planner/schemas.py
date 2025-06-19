from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class CalculateCompInput(BaseModel):
    role: str = Field(..., description="Job title or role")
    level: Optional[str] = Field(None, description="Job level (junior, mid, senior, etc)")
    location: Optional[str] = Field(None, description="Work location")
    company_stage: Optional[str] = Field(None, description="Company stage (startup, series A, etc)")
    special_requirements: Optional[str] = Field(None, description="Special requirements or constraints")

    @validator('role')
    def role_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Role cannot be empty')
        return v

class CompensationOutput(BaseModel):
    base_salary_range: Dict[str, int] = Field(..., description="Min/max base salary")
    bonus_range: Optional[Dict[str, int]] = Field(None, description="Min/max bonus")
    equity_value: Optional[int] = Field(None, description="Annual equity value")
    benefits: Optional[List[str]] = Field(None, description="Benefits included")
    total_comp_range: Dict[str, int] = Field(..., description="Min/max total comp")
    confidence_score: float = Field(..., description="Confidence score 0-10")
    justification: str = Field(..., description="Justification for recommendation")

class PolicyCheckInput(BaseModel):
    policy: str = Field(..., description="Policy to check against")
    context: Optional[Dict[str, str]] = Field(None, description="Context for policy check")

class PolicyCheckOutput(BaseModel):
    policy_check: str = Field(..., description="Result of policy check")
    compliant: bool = Field(..., description="Is recommendation compliant?")
    issues: Optional[List[str]] = Field(None, description="List of issues if not compliant")

class RiskFlagInput(BaseModel):
    context: Dict[str, str] = Field(..., description="Context for risk analysis")

class RiskFlagOutput(BaseModel):
    risk: str = Field(..., description="Risk description")
    severity: str = Field(..., description="Severity level")
    mitigation: Optional[str] = Field(None, description="Suggested mitigation")

class GeneralAnswerInput(BaseModel):
    question: str = Field(..., description="General question")

class GeneralAnswerOutput(BaseModel):
    answer: str = Field(..., description="General answer")
    sources: Optional[List[str]] = Field(None, description="Sources used")