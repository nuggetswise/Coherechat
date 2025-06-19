"""
CrewAI Multi-Agent System for Compensation Planning
Implements the three-agent workflow: Recruitment Manager → HR Director → Hiring Manager
"""
import sys
import os

# Apply SQLite patch first thing - CRITICAL for CrewAI to work
try:
    # First try to import and use pysqlite3
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ SQLite patch applied in crewai_agents.py")
except ImportError:
    # If pysqlite3 is not available, try setting environment variable for ChromaDB
    try:
        os.environ['CHROMA_DB_IMPL'] = 'duckdb+parquet'
        print("ℹ️ Set ChromaDB to use DuckDB backend instead of SQLite")
    except Exception:
        print("⚠️ Failed to apply SQLite patches in crewai_agents.py - this may cause issues")

import streamlit as st
from typing import Dict, Any, List, Optional
import cohere
from pydantic import BaseModel, Field
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Try to import CrewAI components, fallback to mock if not available
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
    print("✅ CrewAI imported successfully!")
except ImportError as e:
    print(f"❌ CrewAI import failed: {str(e)}")
    CREWAI_AVAILABLE = False
    # Create mock classes for when CrewAI is not available
    class BaseTool:
        def __init__(self):
            pass
    
    class Agent:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class Task:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class Crew:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        
        def kickoff(self):
            raise Exception("CrewAI not available - using fallback mode")
    
    class Process:
        sequential = "sequential"


class CompensationContext(BaseModel):
    """Context shared between agents"""
    role: str = ""
    level: str = ""
    location: str = ""
    department: str = ""
    budget_range: Optional[Dict[str, int]] = None
    market_data: Optional[Dict[str, Any]] = None
    company_policies: Optional[Dict[str, Any]] = None
    candidate_info: Optional[Dict[str, Any]] = None


class CohereLanguageModel:
    """Custom LLM wrapper for Cohere that works with CrewAI"""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co_client = cohere_client
        self.model_name = "command-r-plus"
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Execute the language model call"""
        try:
            # Handle different input formats that CrewAI might send
            if isinstance(prompt, list):
                # If prompt is a list of messages, extract the content
                if prompt and isinstance(prompt[0], dict) and 'content' in prompt[0]:
                    prompt = prompt[0]['content']
                else:
                    prompt = str(prompt)
            elif not isinstance(prompt, str):
                prompt = str(prompt)
            
            response = self.co_client.chat(
                model=self.model_name,
                message=prompt,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 1500)
            )
            return response.text
        except Exception as e:
            error_msg = f"Cohere API error: {str(e)}"
            print(error_msg)
            # Return a basic response to prevent complete failure
            return f"I encountered an error while processing this request: {str(e)}. Please try again."
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Alternative method name that some frameworks expect"""
        return self.__call__(prompt, **kwargs)
        
    # Add compatibility methods required by CrewAI
    def invoke(self, prompt: str, **kwargs):
        """Compatibility method for newer LLM interfaces"""
        return self.__call__(prompt, **kwargs)
        
    def complete(self, prompt: str, **kwargs):
        """Another compatibility method for older LLM interfaces"""
        return self.__call__(prompt, **kwargs)
    
    # Add batch method for CrewAI compatibility
    def batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Handle batch requests"""
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
    
    # Add async methods for CrewAI compatibility
    async def ainvoke(self, prompt: str, **kwargs):
        """Async compatibility method"""
        return self.__call__(prompt, **kwargs)
    
    async def agenerate(self, prompt: str, **kwargs):
        """Async generate method"""
        return self.__call__(prompt, **kwargs)


class MarketResearchTool(BaseTool):
    """Tool for gathering market compensation data from real dataset"""
    name: str = "Market Research Tool"
    description: str = "Gathers current market compensation data for specific roles and locations from real dataset"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, role: str, level: str, location: str) -> Dict[str, Any]:
        """Execute market research using real compensation data"""
        try:
            # Load real data
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Compensation Data.csv")
            df = pd.read_csv(data_path)
            
            # Filter data based on criteria
            filtered_df = df.copy()
            
            # Filter by role (fuzzy matching)
            if role:
                role_mask = df['job_title'].str.contains(role, case=False, na=False)
                if role_mask.any():
                    filtered_df = filtered_df[role_mask]
            
            # Filter by level  
            if level:
                level_mask = df['job_level'].str.contains(level, case=False, na=False)
                if level_mask.any():
                    filtered_df = filtered_df[level_mask]
            
            # Filter by location
            if location:
                location_mask = df['location'].str.contains(location, case=False, na=False)
                if location_mask.any():
                    filtered_df = filtered_df[location_mask]
            
            # If no matches, use broader criteria
            if filtered_df.empty:
                filtered_df = df.copy()
            
            # Calculate statistics
            base_salaries = filtered_df['base_salary_usd'].values
            bonuses = filtered_df['bonus_usd'].values
            equity_values = filtered_df['equity_value_usd'].values
            total_comps = base_salaries + bonuses + equity_values
            
            # Calculate percentiles
            p25_base = np.percentile(base_salaries, 25)
            p50_base = np.percentile(base_salaries, 50)
            p75_base = np.percentile(base_salaries, 75)
            p90_base = np.percentile(base_salaries, 90)
            
            # Calculate bonus and equity percentages
            avg_bonus_pct = (np.mean(bonuses) / np.mean(base_salaries)) * 100 if np.mean(base_salaries) > 0 else 0
            avg_equity_pct = (np.mean(equity_values) / np.mean(base_salaries)) * 100 if np.mean(base_salaries) > 0 else 0
            
            return {
                "base_salary_range": {
                    "min": int(np.min(base_salaries)),
                    "max": int(np.max(base_salaries)),
                    "avg": int(np.mean(base_salaries))
                },
                "bonus_statistics": {
                    "avg_amount": int(np.mean(bonuses)),
                    "avg_percentage": round(avg_bonus_pct, 1)
                },
                "equity_statistics": {
                    "avg_amount": int(np.mean(equity_values)),
                    "avg_percentage": round(avg_equity_pct, 1)
                },
                "market_percentiles": {
                    "p25": int(p25_base),
                    "p50": int(p50_base),
                    "p75": int(p75_base),
                    "p90": int(p90_base)
                },
                "total_compensation": {
                    "avg": int(np.mean(total_comps)),
                    "min": int(np.min(total_comps)),
                    "max": int(np.max(total_comps))
                },
                "data_sources": ["Real compensation dataset"],
                "sample_size": len(filtered_df),
                "confidence": min(0.95, 0.6 + (len(filtered_df) * 0.05))
            }
            
        except Exception as e:
            # Fallback with estimated data based on common patterns
            return {
                "base_salary_range": {"min": 100000, "max": 200000, "avg": 150000},
                "bonus_statistics": {"avg_amount": 20000, "avg_percentage": 15.0},
                "equity_statistics": {"avg_amount": 50000, "avg_percentage": 30.0},
                "market_percentiles": {"p25": 120000, "p50": 150000, "p75": 180000, "p90": 210000},
                "total_compensation": {"avg": 220000, "min": 150000, "max": 300000},
                "data_sources": ["Estimated market data"],
                "sample_size": 0,
                "confidence": 0.5,
                "error": str(e)
            }
    
    # Add compatibility method for CrewAI tools
    def run(self, role: str, level: str, location: str) -> Dict[str, Any]:
        """Forward to _run for compatibility with different CrewAI versions"""
        return self._run(role, level, location)


class PolicyCheckTool(BaseTool):
    """Tool for checking company compensation policies against real market data"""
    name: str = "Policy Check Tool"
    description: str = "Validates compensation packages against company policies and market benchmarks"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, compensation_package: Dict[str, Any], role: str, level: str) -> Dict[str, Any]:
        """Check policy compliance using real data benchmarks"""
        try:
            # Load real data for benchmarking
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Compensation Data.csv")
            df = pd.read_csv(data_path)
            
            # Get market benchmarks for similar roles
            similar_roles = df[
                (df['job_title'].str.contains(role, case=False, na=False)) |
                (df['job_level'].str.contains(level, case=False, na=False))
            ]
            
            if similar_roles.empty:
                similar_roles = df.copy()
            
            # Calculate market benchmarks
            market_p50_base = np.percentile(similar_roles['base_salary_usd'], 50)
            market_p75_base = np.percentile(similar_roles['base_salary_usd'], 75)
            market_p90_base = np.percentile(similar_roles['base_salary_usd'], 90)
            
            avg_total_comp = np.mean(
                similar_roles['base_salary_usd'] + 
                similar_roles['bonus_usd'] + 
                similar_roles['equity_value_usd']
            )
            
            issues = []
            recommendations = []
            policy_score = 10.0
            
            # Extract proposed compensation
            base_salary = compensation_package.get("base_salary", {})
            if isinstance(base_salary, dict):
                salary_amount = base_salary.get("recommended", base_salary.get("max", 0))
            else:
                salary_amount = base_salary
            
            # Market competitiveness check
            if salary_amount < market_p50_base * 0.8:
                issues.append(f"Base salary ${salary_amount:,} is significantly below market median ${market_p50_base:,.0f}")
                policy_score -= 2.0
                recommendations.append("Consider increasing base salary to be competitive")
            elif salary_amount > market_p90_base * 1.2:
                issues.append(f"Base salary ${salary_amount:,} significantly exceeds market 90th percentile ${market_p90_base:,.0f}")
                policy_score -= 1.5
                recommendations.append("Executive approval required for above-market compensation")
            
            # Budget threshold check
            if salary_amount > 250000:
                issues.append("Salary exceeds standard approval threshold - requires VP+ approval")
                policy_score -= 1.0
                
            # Total compensation check
            bonus = compensation_package.get("bonus", {})
            equity = compensation_package.get("equity", {})
            
            bonus_amount = bonus.get("target", 0) if isinstance(bonus, dict) else bonus
            equity_amount = equity.get("estimated_value", 0) if isinstance(equity, dict) else equity
            
            total_proposed = salary_amount + bonus_amount + equity_amount
            
            if total_proposed > avg_total_comp * 1.5:
                issues.append(f"Total compensation ${total_proposed:,} significantly exceeds market average ${avg_total_comp:,.0f}")
                policy_score -= 1.5
            
            # Equity policy check
            if isinstance(equity, dict):
                equity_pct = equity.get("percentage", 0)
                if equity_pct > 1.0:
                    issues.append("Equity percentage exceeds 1% - requires equity committee approval")
                    policy_score -= 0.5
            
            # Approval level determination
            if policy_score >= 9.0:
                approval_level = "manager"
            elif policy_score >= 7.0:
                approval_level = "director"
            elif policy_score >= 5.0:
                approval_level = "vp"
            else:
                approval_level = "executive"
            
            # Generate recommendations if no issues
            if not issues:
                recommendations.extend([
                    "Compensation package is well-aligned with market benchmarks",
                    f"Salary within market range (P50: ${market_p50_base:,.0f})",
                    "Package complies with all standard policies"
                ])
            
            return {
                "compliant": len(issues) == 0,
                "issues": issues,
                "recommendations": recommendations,
                "approval_level": approval_level,
                "policy_score": policy_score,
                "market_benchmarks": {
                    "market_p50_base": market_p50_base,
                    "market_p75_base": market_p75_base,
                    "market_p90_base": market_p90_base,
                    "avg_total_comp": avg_total_comp
                },
                "benchmark_sample_size": len(similar_roles)
            }
            
        except Exception as e:
            return {
                "compliant": False,
                "issues": [f"Policy check failed: {str(e)}"],
                "recommendations": ["Manual review required due to system error"],
                "approval_level": "director",
                "policy_score": 5.0,
                "error": str(e)
            }
    
    # Add compatibility method for CrewAI tools
    def run(self, compensation_package: Dict[str, Any], role: str, level: str) -> Dict[str, Any]:
        """Forward to _run for compatibility with different CrewAI versions"""
        return self._run(compensation_package, role, level)


class RiskAssessmentTool(BaseTool):
    """Tool for assessing compensation risks using market data analysis"""
    name: str = "Risk Assessment Tool"
    description: str = "Identifies potential risks in compensation recommendations using real market data"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, compensation_package: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess compensation risks using real market analysis"""
        try:
            # Load real data for risk analysis
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Compensation Data.csv")
            df = pd.read_csv(data_path)
            
            risks = []
            risk_level = "low"
            risk_score = 2.0
            
            # Extract compensation components
            base_salary = compensation_package.get("base_salary", {})
            if isinstance(base_salary, dict):
                salary_amount = base_salary.get("recommended", base_salary.get("max", 0))
            else:
                salary_amount = base_salary
            
            bonus = compensation_package.get("bonus", {})
            bonus_amount = bonus.get("target", 0) if isinstance(bonus, dict) else bonus
            
            total_comp = salary_amount + bonus_amount
            
            # Calculate market statistics
            market_salaries = df['base_salary_usd'].values
            market_total = df['base_salary_usd'] + df['bonus_usd'] + df['equity_value_usd']
            
            market_p50 = np.percentile(market_salaries, 50)
            market_p90 = np.percentile(market_salaries, 90)
            market_avg_total = np.mean(market_total)
            
            # Market competitiveness risks
            if salary_amount < market_p50 * 0.85:
                risks.append("Below-market compensation may impact ability to attract and retain top talent")
                risk_level = "medium"
                risk_score += 2.0
            elif salary_amount > market_p90 * 1.2:
                risks.append("Above-market compensation may create internal equity issues and set unsustainable precedents")
                risk_level = "medium"
                risk_score += 1.5
            
            # Budget and financial risks
            if total_comp > market_avg_total * 1.4:
                risks.append("High total compensation may strain department budget and impact profitability")
                risk_level = "high"
                risk_score += 2.5
            
            if salary_amount > 300000:
                risks.append("High base salary creates ongoing fixed cost commitment")
                risk_score += 1.0
                if risk_level == "low":
                    risk_level = "medium"
            
            # Retention and motivation risks
            rejected_offers = df[df['offer_outcome'] == 'Rejected']
            if not rejected_offers.empty:
                avg_rejected_salary = np.mean(rejected_offers['base_salary_usd'])
                if salary_amount <= avg_rejected_salary:
                    risks.append("Compensation level similar to previously rejected offers - high rejection risk")
                    risk_score += 1.5
                    if risk_level == "low":
                        risk_level = "medium"
            
            # Market trend analysis
            high_equity_preference = df[df['candidate_preference'] == 'More Equity']
            if len(high_equity_preference) > len(df) * 0.3:  # More than 30% prefer equity
                equity = compensation_package.get("equity", {})
                if isinstance(equity, dict):
                    equity_amount = equity.get("estimated_value", 0)
                    if equity_amount < salary_amount * 0.2:
                        risks.append("Low equity component may not align with current market preferences")
                        risk_score += 1.0
            
            # Generate mitigation strategies
            mitigation_strategies = [
                "Regular market benchmarking and compensation reviews",
                "Performance-based compensation adjustments",
                "Retention monitoring and stay interviews"
            ]
            
            if "equity" in str(risks).lower():
                mitigation_strategies.append("Consider equity refresh grants or performance bonuses")
            
            if "budget" in str(risks).lower():
                mitigation_strategies.append("Implement performance milestones tied to compensation")
            
            if "market" in str(risks).lower():
                mitigation_strategies.append("Benchmark against role-specific market data quarterly")
            
            # Final risk level determination
            if risk_score >= 6.0:
                risk_level = "high"
            elif risk_score >= 3.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "risk_level": risk_level,
                "risks": risks if risks else ["No significant compensation risks identified"],
                "mitigation_strategies": mitigation_strategies,
                "risk_score": round(risk_score, 1),
                "market_comparison": {
                    "vs_market_median": f"{((salary_amount / market_p50 - 1) * 100):+.1f}%",
                    "vs_market_p90": f"{((salary_amount / market_p90 - 1) * 100):+.1f}%",
                    "total_comp_vs_avg": f"{((total_comp / market_avg_total - 1) * 100):+.1f}%"
                },
                "data_insights": {
                    "rejection_rate": f"{len(rejected_offers) / len(df) * 100:.1f}%",
                    "avg_rejected_salary": f"${np.mean(rejected_offers['base_salary_usd']) if not rejected_offers.empty else 0:,.0f}",
                    "market_sample_size": len(df)
                }
            }
            
        except Exception as e:
            return {
                "risk_level": "medium",
                "risks": [f"Risk assessment failed: {str(e)}", "Manual risk review recommended"],
                "mitigation_strategies": [
                    "Conduct thorough manual review",
                    "Benchmark against multiple data sources",
                    "Consider conservative approach"
                ],
                "risk_score": 5.0,
                "error": str(e)
            }
    
    # Add compatibility method for CrewAI tools
    def run(self, compensation_package: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Forward to _run for compatibility with different CrewAI versions"""
        return self._run(compensation_package, market_data)


class CompensationCrewAI:
    """Main CrewAI orchestrator for compensation planning"""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co_client = cohere_client
        self.cohere_llm = CohereLanguageModel(cohere_client)
        self.tools = self._setup_tools()
        self.crewai_available = CREWAI_AVAILABLE
        
        if self.crewai_available:
            self.agents = self._create_agents()
        else:
            self.agents = {}
        
    def _setup_tools(self) -> List[BaseTool]:
        """Setup available tools for agents"""
        return [
            MarketResearchTool(),
            PolicyCheckTool(),
            RiskAssessmentTool()
        ]
        
    def _create_agents(self) -> Dict[str, Agent]:
        """Create the three specialized agents"""
        
        if not self.crewai_available:
            return {}
        
        # Import persona prompts
        from comp_planner.persona_prompts import get_persona_prompt
        
        # Recruitment Manager Agent
        recruitment_manager = Agent(
            role="Recruitment Manager",
            goal="Draft competitive compensation offers that attract top talent while being cost-effective",
            backstory=get_persona_prompt("recruitment_manager", "system_prompt", role="", level="", location=""),
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            llm=self.cohere_llm  # Use our Cohere LLM wrapper
        )
        
        # HR Director Agent  
        hr_director = Agent(
            role="HR Director",
            goal="Validate compensation packages for policy compliance, internal equity, and long-term sustainability",
            backstory=get_persona_prompt("hr_director", "system_prompt", role="", level="", department=""),
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            llm=self.cohere_llm  # Use our Cohere LLM wrapper
        )
        
        # Hiring Manager Agent
        hiring_manager = Agent(
            role="Hiring Manager", 
            goal="Make final approval decisions on compensation packages based on team needs and budget constraints",
            backstory=get_persona_prompt("hiring_manager", "system_prompt"),
            tools=self.tools,
            verbose=True,
            allow_delegation=False,
            llm=self.cohere_llm  # Use our Cohere LLM wrapper
        )
        
        return {
            "recruitment_manager": recruitment_manager,
            "hr_director": hr_director, 
            "hiring_manager": hiring_manager
        }
    
    def create_compensation_workflow(self, context: CompensationContext) -> List[Task]:
        """Create the three-stage compensation workflow"""
        
        if not self.crewai_available:
            return []
        
        # Task 1: Recruitment Manager drafts initial offer
        draft_offer_task = Task(
            description=f"""
            Draft a comprehensive compensation offer for a {context.level} {context.role} position in {context.location}.
            
            Requirements:
            - Research current market rates using available tools
            - Create a competitive base salary recommendation
            - Include bonus and equity components
            - Provide justification for each component
            - Consider candidate appeal and market positioning
            
            Context: {context.dict()}
            
            Output: A detailed compensation package with market analysis and rationale.
            """,
            agent=self.agents["recruitment_manager"],
            expected_output="Compensation package with base salary, bonus, equity, and market justification"
        )
        
        # Task 2: HR Director validates policy compliance
        policy_validation_task = Task(
            description=f"""
            Review the compensation package drafted by the Recruitment Manager for policy compliance and internal equity.
            
            Requirements:
            - Validate against company compensation policies
            - Check for internal equity considerations
            - Assess long-term sustainability and precedent implications
            - Identify any required approvals or exceptions
            - Provide recommendations for adjustments if needed
            
            Input: Use the compensation package from the previous task
            
            Output: Policy compliance assessment with any recommended modifications.
            """,
            agent=self.agents["hr_director"],
            expected_output="Policy validation report with compliance status and modification recommendations"
        )
        
        # Task 3: Hiring Manager makes final approval decision
        final_approval_task = Task(
            description=f"""
            Make the final decision on the compensation package after reviewing the draft offer and policy validation.
            
            Requirements:
            - Consider team budget constraints and departmental needs
            - Evaluate the business case for the proposed compensation
            - Make final adjustments based on practical considerations
            - Provide clear approval decision with rationale
            - Include any conditions or contingencies
            
            Input: Use both the initial compensation package and the policy validation results
            
            Output: Final approved compensation package with decision rationale.
            """,
            agent=self.agents["hiring_manager"],
            expected_output="Final compensation decision with approved package and business justification"
        )
        
        return [draft_offer_task, policy_validation_task, final_approval_task]
    
    def execute_compensation_planning(self, context: CompensationContext) -> Dict[str, Any]:
        """Execute the full compensation planning workflow"""
        
        # If CrewAI is not available, use fallback immediately
        if not self.crewai_available:
            return {
                "success": False,
                "error": "CrewAI not available - using fallback mode",
                "context": context.dict(),
                "fallback_recommendation": self._generate_fallback_recommendation(context)
            }
        
        # Create tasks for the workflow
        tasks = self.create_compensation_workflow(context)
        
        # Create and configure the crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential,  # Sequential execution: RM → HR → HM
            verbose=True
        )
        
        try:
            # Execute the crew workflow
            result = crew.kickoff()
            
            return {
                "success": True,
                "workflow_result": result,
                "tasks_completed": len(tasks),
                "agents_involved": list(self.agents.keys()),
                "execution_time": datetime.now().isoformat(),
                "context": context.dict()
            }
            
        except Exception as e:
            # Enhanced error handling with more context
            error_msg = str(e)
            
            return {
                "success": False,
                "error": error_msg,
                "error_type": "workflow_execution_failed",
                "context": context.dict(),
                "fallback_recommendation": self._generate_fallback_recommendation(context),
                "debug_info": {
                    "crewai_available": self.crewai_available,
                    "agents_count": len(self.agents),
                    "tools_count": len(self.tools),
                    "cohere_client_available": bool(self.co_client)
                }
            }
    
    def _generate_fallback_recommendation(self, context: CompensationContext) -> Dict[str, Any]:
        """Generate a comprehensive fallback recommendation using Cohere directly"""
        try:
            # Use the existing tools directly for market data
            market_tool = MarketResearchTool()
            market_data = market_tool._run(context.role, context.level, context.location)
            
            # Generate a detailed recommendation using Cohere
            prompt = f"""
            As an expert compensation consultant, create a comprehensive compensation package for:
            
            Role: {context.level} {context.role}
            Location: {context.location}
            Department: {context.department}
            
            Market Data Available:
            - Base Salary Range: ${market_data['base_salary_range']['min']:,} - ${market_data['base_salary_range']['max']:,}
            - Market P50: ${market_data['market_percentiles']['p50']:,}
            - Bonus: Average ${market_data['bonus_statistics']['avg_amount']:,} ({market_data['bonus_statistics']['avg_percentage']}% of base)
            - Equity: Average ${market_data['equity_statistics']['avg_amount']:,} ({market_data['equity_statistics']['avg_percentage']}% of base)
            
            Provide a detailed compensation recommendation including:
            1. Recommended base salary with justification
            2. Bonus structure and target amount
            3. Equity recommendation
            4. Total compensation summary
            5. Market competitiveness analysis
            6. Key considerations and risks
            
            Format as a professional compensation recommendation.
            """
            
            response = self.co_client.chat(
                model="command-r-plus",
                message=prompt,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Calculate actual recommended values based on market data, not hardcoded values
            base_salary = market_data["base_salary_range"]
            p50 = market_data["market_percentiles"]["p50"]
            avg = base_salary["avg"]
            
            # Use p50 if available, otherwise use the average of min and max
            if p50 > 0:
                recommended_salary = int(p50)
            else:
                recommended_salary = int(avg)
            
            # Calculate bonus percentage from actual data
            bonus_percentage = market_data["bonus_statistics"]["avg_percentage"]
            bonus_amount = int(recommended_salary * (bonus_percentage / 100))
            
            # Calculate equity percentage from actual data
            equity_percentage = market_data["equity_statistics"]["avg_percentage"] / 100
            equity_amount = int(recommended_salary * (equity_percentage))
            
            # Calculate total comp based on real values
            total_comp = recommended_salary + bonus_amount + equity_amount
            
            return {
                "recommendation": response.text,
                "base_salary": {
                    "recommended": recommended_salary,
                    "range": base_salary
                },
                "bonus": {
                    "target": bonus_amount,
                    "range": f"{max(bonus_percentage - 5, 0):.1f}-{bonus_percentage + 5:.1f}% of base salary"
                },
                "equity": {
                    "percentage": equity_percentage,
                    "description": f"{max(equity_percentage - 0.1, 0):.1%}-{equity_percentage + 0.1:.1%} equity over 4 years",
                    "estimated_value": equity_amount
                },
                "total_compensation": {
                    "estimated": total_comp,
                    "components": "Base + Bonus + Equity + Benefits"
                },
                "market_data": market_data,
                "justification": f"Market-competitive package based on real data for {context.level} {context.role} in {context.location}",
                "approval_status": "pending",
                "source": "cohere_enhanced_fallback",
                "confidence_score": 8.5
            }
            
        except Exception as e:
            # Read actual data from CSV for ultra-fallback
            try:
                import pandas as pd
                data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Compensation Data.csv")
                df = pd.read_csv(data_path)
                
                # Filter by role and level if possible
                filtered_df = df.copy()
                if context.role:
                    role_mask = df['job_title'].str.contains(context.role, case=False, na=False)
                    if role_mask.any():
                        filtered_df = filtered_df[role_mask]
                
                if context.level:
                    level_mask = df['job_level'].str.contains(context.level, case=False, na=False)
                    if level_mask.any():
                        filtered_df = filtered_df[level_mask]
                
                if filtered_df.empty:
                    filtered_df = df
                
                # Get average values
                avg_base = filtered_df['base_salary_usd'].mean()
                avg_bonus = filtered_df['bonus_usd'].mean()
                avg_equity = filtered_df['equity_value_usd'].mean()
                
                min_base = filtered_df['base_salary_usd'].min()
                max_base = filtered_df['base_salary_usd'].max()
                
                # Calculate percentages
                bonus_pct = (avg_bonus / avg_base * 100) if avg_base > 0 else 15
                equity_pct = (avg_equity / avg_base * 100) if avg_base > 0 else 25
                equity_decimal = equity_pct / 100
                
                # Create recommendation
                recommended_salary = int(avg_base)
                bonus_amount = int(avg_bonus)
                equity_amount = int(avg_equity)
                total_comp = recommended_salary + bonus_amount + equity_amount
                
                return {
                    "recommendation": f"Based on market analysis of real compensation data, we recommend a compensation package for {context.level} {context.role} in {context.location} with a base salary of ${recommended_salary:,}, annual bonus target of ${bonus_amount:,} ({bonus_pct:.1f}% of base), and equity grant of {equity_decimal:.3f}%. This represents a competitive total compensation of approximately ${total_comp:,}.",
                    "base_salary": {
                        "recommended": recommended_salary,
                        "range": {
                            "min": int(min_base),
                            "max": int(max_base),
                            "avg": int(avg_base)
                        }
                    },
                    "bonus": {
                        "target": bonus_amount,
                        "range": f"{max(bonus_pct - 5, 0):.1f}-{bonus_pct + 5:.1f}% of base salary"
                    },
                    "equity": {
                        "percentage": equity_decimal,
                        "description": f"{max(equity_decimal - 0.1, 0):.1%}-{equity_decimal + 0.1:.1%} equity over 4 years",
                        "estimated_value": equity_amount
                    },
                    "total_compensation": {
                        "estimated": total_comp,
                        "components": "Base + Bonus + Equity + Benefits"
                    },
                    "justification": f"Data-driven package based on {len(filtered_df)} real compensation records for {context.level} {context.role} in {context.location}",
                    "approval_status": "pending",
                    "source": "csv_data_fallback",
                    "error": str(e),
                    "confidence_score": 7.0
                }
                
            except Exception as nested_e:
                # Last resort fallback with empty data
                return {
                    "recommendation": f"Unable to generate recommendation for {context.level} {context.role} in {context.location} due to data access errors. Please check the data source and try again.",
                    "base_salary": {
                        "recommended": 0,
                        "range": {"min": 0, "max": 0, "avg": 0}
                    },
                    "bonus": {
                        "target": 0,
                        "range": "N/A"
                    },
                    "equity": {
                        "percentage": 0,
                        "description": "N/A",
                        "estimated_value": 0
                    },
                    "total_compensation": {
                        "estimated": 0,
                        "components": "Error: No data available"
                    },
                    "justification": "Error: Could not access compensation data",
                    "approval_status": "error",
                    "source": "error_fallback",
                    "error": f"Primary error: {str(e)}. Fallback error: {str(nested_e)}",
                    "confidence_score": 0.0
                }


def get_crewai_compensation_planner(cohere_client: cohere.Client) -> CompensationCrewAI:
    """Factory function to create CrewAI compensation planner"""
    return CompensationCrewAI(cohere_client)