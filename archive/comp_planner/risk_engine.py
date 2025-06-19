"""
Risk Assessment & Policy Engine for Compensation Planning.
Detects budget violations, policy compliance issues, and market risks.
"""
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import cohere

class RiskLevel(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskType(str, Enum):
    """Types of risks that can be detected."""
    BUDGET_VIOLATION = "budget_violation"
    POLICY_VIOLATION = "policy_violation"
    MARKET_OUTLIER = "market_outlier"
    EQUITY_DILUTION = "equity_dilution"
    COMPRESSION_RISK = "compression_risk"
    RETENTION_RISK = "retention_risk"
    LEGAL_COMPLIANCE = "legal_compliance"

class Risk(BaseModel):
    """Individual risk detection."""
    risk_id: str
    risk_type: RiskType
    risk_level: RiskLevel
    description: str
    impact: str
    recommendation: str
    confidence: float
    detected_at: datetime
    affected_components: List[str] = []

class PolicyRule(BaseModel):
    """Company policy rule for compensation."""
    rule_id: str
    rule_type: str
    description: str
    threshold: Optional[float] = None
    condition: str
    violation_message: str
    severity: RiskLevel

class RiskEngine:
    """Advanced risk assessment and policy compliance engine."""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co_client = cohere_client
        self.policy_rules = self._initialize_default_policies()
        self.risk_history = []
        
    def _initialize_default_policies(self) -> List[PolicyRule]:
        """Initialize common compensation policy rules."""
        return [
            PolicyRule(
                rule_id="salary_band_compliance",
                rule_type="salary_range",
                description="Salary must be within approved band for level",
                condition="salary_within_band",
                violation_message="Proposed salary exceeds approved band for this level",
                severity=RiskLevel.HIGH
            ),
            PolicyRule(
                rule_id="equity_dilution_limit",
                rule_type="equity",
                description="Equity grants should not exceed dilution limits",
                threshold=0.5,
                condition="equity_percentage_limit",
                violation_message="Equity grant may cause excessive dilution",
                severity=RiskLevel.MEDIUM
            ),
            PolicyRule(
                rule_id="total_comp_ceiling",
                rule_type="total_compensation",
                description="Total compensation should not exceed budget ceiling",
                condition="total_comp_limit",
                violation_message="Total compensation exceeds approved budget ceiling",
                severity=RiskLevel.CRITICAL
            ),
            PolicyRule(
                rule_id="compression_prevention",
                rule_type="internal_equity",
                description="New hire salary should not compress existing team",
                condition="no_compression",
                violation_message="Proposed salary may compress existing team members",
                severity=RiskLevel.HIGH
            ),
            PolicyRule(
                rule_id="market_reasonableness",
                rule_type="market_alignment",
                description="Compensation should align with market standards",
                condition="market_reasonable",
                violation_message="Compensation significantly deviates from market",
                severity=RiskLevel.MEDIUM
            )
        ]
    
    def assess_compensation_risks(self, 
                                 recommendation: Dict[str, Any], 
                                 context: Dict[str, Any] = None,
                                 company_policies: Dict[str, Any] = None) -> List[Risk]:
        """Comprehensive risk assessment of compensation recommendation."""
        risks = []
        
        # Extract compensation components
        comp_data = self._parse_compensation_data(recommendation)
        
        # Budget violation checks
        budget_risks = self._check_budget_violations(comp_data, context)
        risks.extend(budget_risks)
        
        # Policy compliance checks
        policy_risks = self._check_policy_compliance(comp_data, company_policies)
        risks.extend(policy_risks)
        
        # Market outlier detection
        market_risks = self._detect_market_outliers(comp_data, context)
        risks.extend(market_risks)
        
        # Internal equity risks
        equity_risks = self._assess_internal_equity_risks(comp_data, context)
        risks.extend(equity_risks)
        
        # Legal and compliance risks
        legal_risks = self._check_legal_compliance(comp_data, context)
        risks.extend(legal_risks)
        
        # Store risk history
        for risk in risks:
            self.risk_history.append(risk)
        
        return risks
    
    def _parse_compensation_data(self, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured compensation data from recommendation."""
        rec_text = recommendation.get('recommendation', '')
        
        # Use Cohere to extract structured data
        prompt = f"""
        Extract compensation details from this recommendation text:
        
        "{rec_text}"
        
        Extract and return in JSON format:
        {{
            "base_salary": {{
                "min": 150000,
                "max": 180000,
                "currency": "USD"
            }},
            "bonus": {{
                "target_percentage": 20,
                "amount": 30000
            }},
            "equity": {{
                "value": 100000,
                "percentage": 0.25,
                "vesting_years": 4
            }},
            "total_compensation": {{
                "min": 280000,
                "max": 310000
            }},
            "location": "San Francisco",
            "level": "Senior",
            "role": "Software Engineer"
        }}
        
        If specific numbers aren't mentioned, use null values.
        """
        
        try:
            response = self.co_client.generate(prompt=prompt, max_tokens=400, temperature=0.1)
            comp_text = response.generations[0].text.strip()
            
            start_idx = comp_text.find('{')
            end_idx = comp_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = comp_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback parsing
        return {
            "base_salary": {"min": None, "max": None},
            "bonus": {"target_percentage": None},
            "equity": {"value": None, "percentage": None},
            "total_compensation": {"min": None, "max": None}
        }
    
    def _check_budget_violations(self, comp_data: Dict[str, Any], context: Dict[str, Any] = None) -> List[Risk]:
        """Check for budget violations."""
        risks = []
        
        # Extract budget constraints from context
        budget_max = None
        if context and 'budget_constraints' in context:
            budget_max = context['budget_constraints'].get('max_total_comp')
        
        # Check total compensation against budget
        total_comp = comp_data.get('total_compensation', {})
        if total_comp.get('max') and budget_max:
            if total_comp['max'] > budget_max:
                risks.append(Risk(
                    risk_id=f"budget_violation_{int(datetime.now().timestamp())}",
                    risk_type=RiskType.BUDGET_VIOLATION,
                    risk_level=RiskLevel.CRITICAL,
                    description=f"Recommended total compensation ({total_comp['max']:,}) exceeds budget ceiling ({budget_max:,})",
                    impact="Recommendation cannot be approved without budget adjustment",
                    recommendation="Reduce total compensation or request budget increase",
                    confidence=0.95,
                    detected_at=datetime.now(),
                    affected_components=["total_compensation"]
                ))
        
        # Check if salary is suspiciously high
        base_salary = comp_data.get('base_salary', {})
        if base_salary.get('max') and base_salary['max'] > 500000:
            risks.append(Risk(
                risk_id=f"high_salary_{int(datetime.now().timestamp())}",
                risk_type=RiskType.BUDGET_VIOLATION,
                risk_level=RiskLevel.MEDIUM,
                description=f"Base salary ({base_salary['max']:,}) is exceptionally high",
                impact="May indicate data error or require executive approval",
                recommendation="Verify salary level is intentional and within company guidelines",
                confidence=0.7,
                detected_at=datetime.now(),
                affected_components=["base_salary"]
            ))
        
        return risks
    
    def _check_policy_compliance(self, comp_data: Dict[str, Any], company_policies: Dict[str, Any] = None) -> List[Risk]:
        """Check compliance with company policies."""
        risks = []
        
        # Use default policies if none provided
        policies = company_policies or {}
        
        # Check salary band compliance
        role = comp_data.get('role', '').lower()
        level = comp_data.get('level', '').lower()
        base_salary = comp_data.get('base_salary', {})
        
        if 'salary_bands' in policies and base_salary.get('max'):
            band_key = f"{role}_{level}"
            if band_key in policies['salary_bands']:
                band = policies['salary_bands'][band_key]
                if base_salary['max'] > band.get('max', float('inf')):
                    risks.append(Risk(
                        risk_id=f"salary_band_violation_{int(datetime.now().timestamp())}",
                        risk_type=RiskType.POLICY_VIOLATION,
                        risk_level=RiskLevel.HIGH,
                        description=f"Salary exceeds approved band maximum ({band['max']:,})",
                        impact="Violates company compensation guidelines",
                        recommendation="Adjust salary to within approved band or request exception",
                        confidence=0.9,
                        detected_at=datetime.now(),
                        affected_components=["base_salary"]
                    ))
        
        # Check equity dilution
        equity = comp_data.get('equity', {})
        if equity.get('percentage') and equity['percentage'] > 1.0:  # >1% equity
            risks.append(Risk(
                risk_id=f"equity_dilution_{int(datetime.now().timestamp())}",
                risk_type=RiskType.EQUITY_DILUTION,
                risk_level=RiskLevel.MEDIUM,
                description=f"Equity grant ({equity['percentage']:.2%}) is substantial",
                impact="May require board approval due to dilution impact",
                recommendation="Review equity grant with compensation committee",
                confidence=0.8,
                detected_at=datetime.now(),
                affected_components=["equity"]
            ))
        
        return risks
    
    def _detect_market_outliers(self, comp_data: Dict[str, Any], context: Dict[str, Any] = None) -> List[Risk]:
        """Detect compensation that significantly deviates from market."""
        risks = []
        
        # Get market data from context or use AI to assess
        total_comp = comp_data.get('total_compensation', {})
        location = comp_data.get('location', '')
        role = comp_data.get('role', '')
        level = comp_data.get('level', '')
        
        if total_comp.get('max'):
            # Use AI to assess market reasonableness
            market_assessment = self._assess_market_alignment(comp_data)
            
            if market_assessment.get('deviation_severity') == 'high':
                risks.append(Risk(
                    risk_id=f"market_outlier_{int(datetime.now().timestamp())}",
                    risk_type=RiskType.MARKET_OUTLIER,
                    risk_level=RiskLevel.MEDIUM,
                    description=market_assessment.get('description', 'Compensation deviates significantly from market'),
                    impact="May indicate overpayment or underpayment relative to market",
                    recommendation=market_assessment.get('recommendation', 'Review market data and adjust if necessary'),
                    confidence=market_assessment.get('confidence', 0.6),
                    detected_at=datetime.now(),
                    affected_components=["total_compensation"]
                ))
        
        return risks
    
    def _assess_market_alignment(self, comp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to assess market alignment."""
        prompt = f"""
        Assess if this compensation package aligns with market standards:
        
        Role: {comp_data.get('role', 'Unknown')}
        Level: {comp_data.get('level', 'Unknown')}
        Location: {comp_data.get('location', 'Unknown')}
        Total Compensation: {comp_data.get('total_compensation', {}).get('max', 'Unknown')}
        Base Salary: {comp_data.get('base_salary', {}).get('max', 'Unknown')}
        
        Provide assessment in JSON:
        {{
            "market_alignment": "low|medium|high",
            "deviation_severity": "low|medium|high",
            "description": "explanation of deviation",
            "recommendation": "suggested action",
            "confidence": 0.7
        }}
        """
        
        try:
            response = self.co_client.generate(prompt=prompt, max_tokens=200, temperature=0.1)
            assessment_text = response.generations[0].text.strip()
            
            start_idx = assessment_text.find('{')
            end_idx = assessment_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = assessment_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "market_alignment": "medium",
            "deviation_severity": "low",
            "description": "Unable to assess market alignment",
            "recommendation": "Manually verify against market data",
            "confidence": 0.3
        }
    
    def _assess_internal_equity_risks(self, comp_data: Dict[str, Any], context: Dict[str, Any] = None) -> List[Risk]:
        """Assess risks to internal pay equity."""
        risks = []
        
        # Check for compression risks
        if context and 'team_compensation' in context:
            team_comp = context['team_compensation']
            proposed_salary = comp_data.get('base_salary', {}).get('max')
            
            if proposed_salary:
                # Check if new hire salary would compress existing team
                for team_member in team_comp:
                    if (team_member.get('level_seniority', 0) > 0 and 
                        team_member.get('current_salary', 0) < proposed_salary):
                        
                        risks.append(Risk(
                            risk_id=f"compression_risk_{int(datetime.now().timestamp())}",
                            risk_type=RiskType.COMPRESSION_RISK,
                            risk_level=RiskLevel.HIGH,
                            description=f"New hire salary ({proposed_salary:,}) may compress {team_member.get('name', 'team member')} ({team_member.get('current_salary', 0):,})",
                            impact="Could create internal equity issues and retention risk",
                            recommendation="Consider adjusting offer or addressing existing team member compensation",
                            confidence=0.8,
                            detected_at=datetime.now(),
                            affected_components=["base_salary"]
                        ))
        
        return risks
    
    def _check_legal_compliance(self, comp_data: Dict[str, Any], context: Dict[str, Any] = None) -> List[Risk]:
        """Check for legal and regulatory compliance issues."""
        risks = []
        
        location = comp_data.get('location', '').lower()
        
        # Check minimum wage compliance (basic check)
        base_salary = comp_data.get('base_salary', {}).get('min', 0)
        if base_salary and base_salary < 31200:  # Federal minimum wage equivalent
            risks.append(Risk(
                risk_id=f"minimum_wage_{int(datetime.now().timestamp())}",
                risk_type=RiskType.LEGAL_COMPLIANCE,
                risk_level=RiskLevel.CRITICAL,
                description="Proposed salary may be below minimum wage requirements",
                impact="Legal compliance violation",
                recommendation="Ensure salary meets local minimum wage laws",
                confidence=0.9,
                detected_at=datetime.now(),
                affected_components=["base_salary"]
            ))
        
        # Location-specific compliance checks
        if 'california' in location or 'san francisco' in location:
            # California salary transparency requirements
            if not comp_data.get('salary_range_disclosed', False):
                risks.append(Risk(
                    risk_id=f"ca_transparency_{int(datetime.now().timestamp())}",
                    risk_type=RiskType.LEGAL_COMPLIANCE,
                    risk_level=RiskLevel.MEDIUM,
                    description="California requires salary range disclosure in job postings",
                    impact="Legal compliance requirement",
                    recommendation="Ensure salary range is disclosed per CA SB 1162",
                    confidence=0.8,
                    detected_at=datetime.now(),
                    affected_components=["disclosure"]
                ))
        
        return risks
    
    def get_risk_summary(self, risks: List[Risk]) -> Dict[str, Any]:
        """Generate summary of risk assessment."""
        if not risks:
            return {
                "total_risks": 0,
                "risk_level": "low",
                "approval_recommendation": "proceed",
                "summary": "No significant risks detected"
            }
        
        risk_counts = {}
        for level in RiskLevel:
            risk_counts[level.value] = len([r for r in risks if r.risk_level == level])
        
        # Determine overall risk level
        if risk_counts.get('critical', 0) > 0:
            overall_risk = "critical"
            approval_rec = "requires_executive_approval"
        elif risk_counts.get('high', 0) > 0:
            overall_risk = "high"
            approval_rec = "requires_approval"
        elif risk_counts.get('medium', 0) > 0:
            overall_risk = "medium"
            approval_rec = "proceed_with_caution"
        else:
            overall_risk = "low"
            approval_rec = "proceed"
        
        return {
            "total_risks": len(risks),
            "risk_counts": risk_counts,
            "risk_level": overall_risk,
            "approval_recommendation": approval_rec,
            "summary": f"Detected {len(risks)} risks: {', '.join([f'{count} {level}' for level, count in risk_counts.items() if count > 0])}",
            "critical_risks": [r.description for r in risks if r.risk_level == RiskLevel.CRITICAL],
            "high_risks": [r.description for r in risks if r.risk_level == RiskLevel.HIGH]
        }
    
    def add_custom_policy(self, policy: PolicyRule):
        """Add a custom company policy rule."""
        self.policy_rules.append(policy)
    
    def get_risk_history(self) -> List[Risk]:
        """Get historical risk data for learning."""
        return self.risk_history