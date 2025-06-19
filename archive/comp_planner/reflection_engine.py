"""
Self-Reflection and Correction Engine for Agentic Compensation Planning.
Validates recommendations, detects errors, and implements self-correction.
"""
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import cohere
from dataclasses import dataclass

class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"

class IssueType(str, Enum):
    """Types of issues that can be detected."""
    LOGICAL_ERROR = "logical_error"
    DATA_INCONSISTENCY = "data_inconsistency"
    POLICY_VIOLATION = "policy_violation"
    MARKET_MISMATCH = "market_mismatch"
    CALCULATION_ERROR = "calculation_error"
    CONFIDENCE_LOW = "confidence_low"
    INCOMPLETE_INFO = "incomplete_info"

@dataclass
class ValidationIssue:
    """Represents a validation issue found during reflection."""
    issue_type: IssueType
    severity: str  # low, medium, high, critical
    description: str
    suggested_fix: str
    confidence: float
    affected_fields: List[str]

@dataclass
class ReflectionResult:
    """Result of reflection analysis."""
    overall_confidence: float
    issues_found: List[ValidationIssue]
    corrected_output: Dict[str, Any]
    improvement_suggestions: List[str]
    validation_passed: bool
    reasoning_chain: List[str]

class ReflectionEngine:
    """Advanced self-reflection and correction system."""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co_client = cohere_client
        self.validation_rules = self._initialize_validation_rules()
        self.correction_history = []
        
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules and criteria."""
        return {
            "salary_ranges": {
                "min_salary": 50000,
                "max_salary": 1000000,
                "percentile_order": ["p25", "p50", "p75", "p90"],
                "logical_progression": True
            },
            "bonus_ranges": {
                "min_percentage": 0.0,
                "max_percentage": 2.0,  # 200% of base
                "typical_range": [0.05, 0.5]
            },
            "equity_ranges": {
                "min_percentage": 0.0,
                "max_percentage": 0.5,  # 50% equity
                "startup_typical": [0.01, 0.1],
                "enterprise_typical": [0.001, 0.05]
            },
            "confidence_thresholds": {
                "minimum_acceptable": 0.6,
                "good": 0.75,
                "excellent": 0.9
            },
            "market_data_requirements": {
                "min_data_points": 3,
                "max_age_days": 365,
                "required_sources": 2
            }
        }
    
    def reflect_and_validate(
        self, 
        recommendation: Dict[str, Any], 
        context: Dict[str, Any],
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ReflectionResult:
        """
        Perform comprehensive reflection and validation on a recommendation.
        """
        issues = []
        reasoning_chain = []
        corrected_output = recommendation.copy()
        
        # Step 1: Basic data validation
        reasoning_chain.append("Starting basic data validation...")
        basic_issues = self._validate_basic_data(recommendation)
        issues.extend(basic_issues)
        
        # Step 2: Logical consistency check
        reasoning_chain.append("Checking logical consistency...")
        logic_issues = self._validate_logical_consistency(recommendation)
        issues.extend(logic_issues)
        
        # Step 3: Market alignment validation
        reasoning_chain.append("Validating market alignment...")
        market_issues = self._validate_market_alignment(recommendation, context)
        issues.extend(market_issues)
        
        # Step 4: Policy compliance check
        reasoning_chain.append("Checking policy compliance...")
        policy_issues = self._validate_policy_compliance(recommendation, context)
        issues.extend(policy_issues)
        
        # Step 5: Confidence assessment
        reasoning_chain.append("Assessing recommendation confidence...")
        confidence_score = self._calculate_confidence(recommendation, context, issues)
        
        # Step 6: Self-correction if needed
        if validation_level in [ValidationLevel.STRICT, ValidationLevel.CRITICAL]:
            reasoning_chain.append("Applying corrections...")
            corrected_output = self._apply_corrections(recommendation, issues)
            
        # Step 7: Generate improvement suggestions
        reasoning_chain.append("Generating improvement suggestions...")
        improvements = self._generate_improvements(recommendation, issues, context)
        
        # Determine if validation passed
        critical_issues = [i for i in issues if i.severity == "critical"]
        validation_passed = len(critical_issues) == 0 and confidence_score >= self.validation_rules["confidence_thresholds"]["minimum_acceptable"]
        
        return ReflectionResult(
            overall_confidence=confidence_score,
            issues_found=issues,
            corrected_output=corrected_output,
            improvement_suggestions=improvements,
            validation_passed=validation_passed,
            reasoning_chain=reasoning_chain
        )
    
    def _validate_basic_data(self, recommendation: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate basic data integrity and completeness."""
        issues = []
        
        # Check required fields
        required_fields = ["base_salary", "total_comp"]
        for field in required_fields:
            if field not in recommendation:
                issues.append(ValidationIssue(
                    issue_type=IssueType.INCOMPLETE_INFO,
                    severity="high",
                    description=f"Missing required field: {field}",
                    suggested_fix=f"Add {field} to recommendation",
                    confidence=0.9,
                    affected_fields=[field]
                ))
        
        # Validate salary ranges
        if "base_salary" in recommendation:
            salary = recommendation["base_salary"]
            if isinstance(salary, dict):
                if "min" in salary and "max" in salary:
                    if salary["min"] > salary["max"]:
                        issues.append(ValidationIssue(
                            issue_type=IssueType.LOGICAL_ERROR,
                            severity="high",
                            description="Minimum salary is greater than maximum salary",
                            suggested_fix="Swap min and max values or recalculate",
                            confidence=0.95,
                            affected_fields=["base_salary"]
                        ))
                    
                    # Check against absolute limits
                    rules = self.validation_rules["salary_ranges"]
                    if salary["min"] < rules["min_salary"]:
                        issues.append(ValidationIssue(
                            issue_type=IssueType.DATA_INCONSISTENCY,
                            severity="medium",
                            description=f"Minimum salary ${salary['min']:,} below market minimum ${rules['min_salary']:,}",
                            suggested_fix=f"Increase minimum to at least ${rules['min_salary']:,}",
                            confidence=0.8,
                            affected_fields=["base_salary"]
                        ))
                    
                    if salary["max"] > rules["max_salary"]:
                        issues.append(ValidationIssue(
                            issue_type=IssueType.DATA_INCONSISTENCY,
                            severity="medium",
                            description=f"Maximum salary ${salary['max']:,} above market maximum ${rules['max_salary']:,}",
                            suggested_fix=f"Cap maximum at ${rules['max_salary']:,}",
                            confidence=0.8,
                            affected_fields=["base_salary"]
                        ))
        
        return issues
    
    def _validate_logical_consistency(self, recommendation: Dict[str, Any]) -> List[ValidationIssue]:
        """Check for logical consistency across all recommendation components."""
        issues = []
        
        # Validate total comp calculation
        if all(k in recommendation for k in ["base_salary", "bonus", "equity", "total_comp"]):
            base = self._extract_value(recommendation["base_salary"])
            bonus = self._extract_value(recommendation["bonus"])
            equity = self._extract_value(recommendation["equity"])
            total_stated = self._extract_value(recommendation["total_comp"])
            
            if all(v is not None for v in [base, bonus, equity, total_stated]):
                calculated_total = base + bonus + equity
                difference = abs(calculated_total - total_stated)
                tolerance = total_stated * 0.05  # 5% tolerance
                
                if difference > tolerance:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.CALCULATION_ERROR,
                        severity="high",
                        description=f"Total compensation mismatch: stated ${total_stated:,} vs calculated ${calculated_total:,}",
                        suggested_fix=f"Correct total compensation to ${calculated_total:,}",
                        confidence=0.9,
                        affected_fields=["total_comp"]
                    ))
        
        # Validate bonus percentage reasonableness
        if "base_salary" in recommendation and "bonus" in recommendation:
            base = self._extract_value(recommendation["base_salary"])
            bonus = self._extract_value(recommendation["bonus"])
            
            if base and bonus:
                bonus_percentage = bonus / base
                rules = self.validation_rules["bonus_ranges"]
                
                if bonus_percentage > rules["max_percentage"]:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.LOGICAL_ERROR,
                        severity="medium",
                        description=f"Bonus {bonus_percentage:.1%} exceeds typical maximum {rules['max_percentage']:.1%}",
                        suggested_fix=f"Consider reducing bonus to {rules['max_percentage']:.1%} of base",
                        confidence=0.7,
                        affected_fields=["bonus"]
                    ))
        
        return issues
    
    def _validate_market_alignment(self, recommendation: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate alignment with market data and benchmarks."""
        issues = []
        
        # Check if recommendation has market data context
        if "market_data" not in context:
            issues.append(ValidationIssue(
                issue_type=IssueType.INCOMPLETE_INFO,
                severity="medium",
                description="No market data provided for validation",
                suggested_fix="Include market benchmarks in recommendation process",
                confidence=0.8,
                affected_fields=["market_data"]
            ))
            return issues
        
        market_data = context["market_data"]
        
        # Validate against market percentiles
        if "salary_ranges" in market_data and "base_salary" in recommendation:
            market_ranges = market_data["salary_ranges"]
            recommended_salary = self._extract_value(recommendation["base_salary"])
            
            if recommended_salary:
                # Check if recommendation falls within reasonable market range
                if "p25" in market_ranges and "p90" in market_ranges:
                    p25 = market_ranges["p25"]
                    p90 = market_ranges["p90"]
                    
                    if recommended_salary < p25 * 0.8:  # 20% below 25th percentile
                        issues.append(ValidationIssue(
                            issue_type=IssueType.MARKET_MISMATCH,
                            severity="medium",
                            description=f"Recommended salary ${recommended_salary:,} significantly below market (P25: ${p25:,})",
                            suggested_fix=f"Consider increasing to at least ${int(p25 * 0.9):,}",
                            confidence=0.75,
                            affected_fields=["base_salary"]
                        ))
                    
                    if recommended_salary > p90 * 1.3:  # 30% above 90th percentile
                        issues.append(ValidationIssue(
                            issue_type=IssueType.MARKET_MISMATCH,
                            severity="high",
                            description=f"Recommended salary ${recommended_salary:,} significantly above market (P90: ${p90:,})",
                            suggested_fix=f"Consider reducing to at most ${int(p90 * 1.1):,}",
                            confidence=0.8,
                            affected_fields=["base_salary"]
                        ))
        
        return issues
    
    def _validate_policy_compliance(self, recommendation: Dict[str, Any], context: Dict[str, Any]) -> List[ValidationIssue]:
        """Check compliance with company policies and constraints."""
        issues = []
        
        # Check budget constraints if provided
        if "budget_constraints" in context:
            constraints = context["budget_constraints"]
            total_comp = self._extract_value(recommendation.get("total_comp"))
            
            if total_comp and "max_total_comp" in constraints:
                max_budget = constraints["max_total_comp"]
                if total_comp > max_budget:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.POLICY_VIOLATION,
                        severity="critical",
                        description=f"Total compensation ${total_comp:,} exceeds budget limit ${max_budget:,}",
                        suggested_fix=f"Reduce total compensation to ${max_budget:,}",
                        confidence=0.95,
                        affected_fields=["total_comp"]
                    ))
        
        # Check equity policy compliance
        if "equity_policy" in context and "equity" in recommendation:
            policy = context["equity_policy"]
            equity_value = self._extract_value(recommendation["equity"])
            
            if equity_value and "max_equity_percentage" in policy:
                # This would need company valuation to validate properly
                # For now, just flag if equity seems excessive
                if "company_valuation" in context:
                    valuation = context["company_valuation"]
                    equity_percentage = equity_value / valuation
                    max_allowed = policy["max_equity_percentage"]
                    
                    if equity_percentage > max_allowed:
                        issues.append(ValidationIssue(
                            issue_type=IssueType.POLICY_VIOLATION,
                            severity="high",
                            description=f"Equity grant {equity_percentage:.2%} exceeds policy limit {max_allowed:.2%}",
                            suggested_fix=f"Reduce equity to {max_allowed:.2%} of company value",
                            confidence=0.85,
                            affected_fields=["equity"]
                        ))
        
        return issues
    
    def _calculate_confidence(
        self, 
        recommendation: Dict[str, Any], 
        context: Dict[str, Any], 
        issues: List[ValidationIssue]
    ) -> float:
        """Calculate overall confidence score for the recommendation."""
        base_confidence = 0.8
        
        # Reduce confidence based on issues
        for issue in issues:
            severity_weights = {"low": 0.02, "medium": 0.05, "high": 0.15, "critical": 0.3}
            weight = severity_weights.get(issue.severity, 0.05)
            base_confidence -= weight
        
        # Adjust based on data quality
        if "market_data" in context:
            market_data = context["market_data"]
            if "data_points" in market_data and market_data["data_points"] >= 10:
                base_confidence += 0.1
            if "sources" in market_data and len(market_data["sources"]) >= 3:
                base_confidence += 0.05
        
        # Adjust based on completeness
        required_fields = ["base_salary", "bonus", "equity", "total_comp"]
        completeness = sum(1 for field in required_fields if field in recommendation) / len(required_fields)
        base_confidence = base_confidence * (0.7 + 0.3 * completeness)
        
        return max(0.0, min(1.0, base_confidence))
    
    def _apply_corrections(self, recommendation: Dict[str, Any], issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Apply automatic corrections for detected issues."""
        corrected = recommendation.copy()
        
        for issue in issues:
            if issue.issue_type == IssueType.CALCULATION_ERROR and "total_comp" in issue.affected_fields:
                # Recalculate total compensation
                base = self._extract_value(corrected.get("base_salary", 0))
                bonus = self._extract_value(corrected.get("bonus", 0))
                equity = self._extract_value(corrected.get("equity", 0))
                
                if all(v is not None for v in [base, bonus, equity]):
                    corrected["total_comp"] = {"calculated": base + bonus + equity}
            
            elif issue.issue_type == IssueType.LOGICAL_ERROR and "base_salary" in issue.affected_fields:
                # Fix min/max swap
                if isinstance(corrected["base_salary"], dict):
                    salary = corrected["base_salary"]
                    if "min" in salary and "max" in salary and salary["min"] > salary["max"]:
                        salary["min"], salary["max"] = salary["max"], salary["min"]
        
        return corrected
    
    def _generate_improvements(
        self, 
        recommendation: Dict[str, Any], 
        issues: List[ValidationIssue], 
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for improving the recommendation."""
        improvements = []
        
        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)
        
        # Generate specific improvements
        if IssueType.MARKET_MISMATCH in issue_types:
            improvements.append("Consider gathering more recent market data for better alignment")
        
        if IssueType.CONFIDENCE_LOW in issue_types:
            improvements.append("Increase data sources and validation checks to improve confidence")
        
        if IssueType.INCOMPLETE_INFO in issue_types:
            improvements.append("Collect additional role and context information for more accurate recommendations")
        
        # Use AI to generate contextual improvements
        if len(issues) > 0:
            try:
                issues_summary = [f"{i.issue_type}: {i.description}" for i in issues[:3]]
                prompt = f"""
                Given these issues with a compensation recommendation:
                {'; '.join(issues_summary)}
                
                Suggest 2-3 specific improvements to make the recommendation more accurate and reliable.
                Focus on actionable suggestions.
                """
                
                response = self.co_client.generate(prompt=prompt, max_tokens=200, temperature=0.3)
                ai_suggestions = response.generations[0].text.strip().split('\n')
                improvements.extend([s.strip('- ').strip() for s in ai_suggestions if s.strip()])
                
            except Exception:
                pass
        
        return list(set(improvements))  # Remove duplicates
    
    def _extract_value(self, field: Any) -> Optional[float]:
        """Extract numeric value from various field formats."""
        if isinstance(field, (int, float)):
            return float(field)
        elif isinstance(field, dict):
            if "value" in field:
                return float(field["value"])
            elif "target" in field:
                return float(field["target"])
            elif "min" in field and "max" in field:
                return (float(field["min"]) + float(field["max"])) / 2
            elif "median" in field:
                return float(field["median"])
        return None
    
    def continuous_learning_from_feedback(self, original_recommendation: Dict[str, Any], 
                                        user_feedback: Dict[str, Any],
                                        final_decision: Dict[str, Any]) -> None:
        """Learn from user feedback to improve future validations."""
        learning_record = {
            "timestamp": datetime.now(),
            "original": original_recommendation,
            "feedback": user_feedback,
            "final": final_decision,
            "accuracy_score": user_feedback.get("accuracy_rating", 0.5)
        }
        
        self.correction_history.append(learning_record)
        
        # Analyze patterns in corrections
        if len(self.correction_history) >= 10:
            self._update_validation_rules()
    
    def _update_validation_rules(self) -> None:
        """Update validation rules based on learning history."""
        # Analyze common correction patterns
        recent_feedback = self.correction_history[-20:]  # Last 20 feedback items
        
        # Look for patterns in salary adjustments
        salary_adjustments = []
        for record in recent_feedback:
            if "base_salary" in record["final"] and "base_salary" in record["original"]:
                original_salary = self._extract_value(record["original"]["base_salary"])
                final_salary = self._extract_value(record["final"]["base_salary"])
                if original_salary and final_salary:
                    adjustment_ratio = final_salary / original_salary
                    salary_adjustments.append(adjustment_ratio)
        
        # Adjust validation thresholds based on patterns
        if salary_adjustments:
            avg_adjustment = sum(salary_adjustments) / len(salary_adjustments)
            if avg_adjustment > 1.1:  # Consistently adjusted upward
                # Relax lower bound validation
                self.validation_rules["salary_ranges"]["min_salary"] *= 0.95
            elif avg_adjustment < 0.9:  # Consistently adjusted downward
                # Tighten upper bound validation
                self.validation_rules["salary_ranges"]["max_salary"] *= 0.95