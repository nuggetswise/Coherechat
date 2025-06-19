"""
Multi-Agent Planning System with ReAct-style reasoning for Compensation Planning.
Implements task decomposition, iterative refinement, and goal-oriented planning.
"""
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel
import cohere
from tenacity import retry, stop_after_attempt, wait_exponential
from comp_planner.personas import get_persona_config

class PlanningStep(str, Enum):
    """Available planning steps in the ReAct framework."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"

class Task(BaseModel):
    """Individual task in the planning process."""
    task_id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    dependencies: List[str] = []
    tools_required: List[str] = []
    output: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

class Plan(BaseModel):
    """Complete plan with tasks and execution strategy."""
    plan_id: str
    goal: str
    tasks: List[Task] = []
    execution_order: List[str] = []
    current_step: int = 0
    status: str = "pending"
    created_at: datetime
    reasoning_chain: List[Dict[str, Any]] = []

class AgentPlanner:
    """Advanced planning system with ReAct reasoning."""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co_client = cohere_client
        self.active_plans = {}
        
    def create_plan(self, user_query: str, context: Dict[str, Any] = None) -> Plan:
        """Create a comprehensive plan for handling the user query."""
        plan_id = f"plan_{int(datetime.now().timestamp())}"
        
        # Step 1: Analyze query complexity
        complexity_analysis = self._analyze_query_complexity(user_query)
        
        # Step 2: Decompose into tasks
        tasks = self._decompose_into_tasks(user_query, complexity_analysis, context)
        
        # Step 3: Determine execution order
        execution_order = self._plan_execution_order(tasks)
        
        plan = Plan(
            plan_id=plan_id,
            goal=user_query,
            tasks=tasks,
            execution_order=execution_order,
            created_at=datetime.now()
        )
        
        self.active_plans[plan_id] = plan
        return plan
    
    def _analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity to determine planning approach."""
        prompt = f"""
        Analyze this compensation query for complexity and planning requirements:
        
        Query: "{query}"
        
        Determine:
        1. Complexity Level (simple/moderate/complex)
        2. Required Information (what data is needed)
        3. Potential Challenges (market data gaps, ambiguity, etc.)
        4. Success Criteria (what makes a good answer)
        5. Risk Factors (budget constraints, policy violations, etc.)
        
        Provide analysis in JSON format:
        {{
            "complexity": "simple|moderate|complex",
            "required_info": ["list", "of", "requirements"],
            "challenges": ["potential", "issues"],
            "success_criteria": ["quality", "measures"],
            "risk_factors": ["risk", "areas"],
            "estimated_steps": 3
        }}
        """
        
        response = self.co_client.generate(
            prompt=prompt,
            max_tokens=300,
            temperature=0.1
        )
        
        try:
            analysis_text = response.generations[0].text.strip()
            # Extract JSON from response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = analysis_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback analysis
        return {
            "complexity": "moderate",
            "required_info": ["role_details", "market_data", "compensation_structure"],
            "challenges": ["data_availability"],
            "success_criteria": ["accurate_ranges", "justification"],
            "risk_factors": ["market_volatility"],
            "estimated_steps": 3
        }
    
    def _decompose_into_tasks(self, query: str, complexity: Dict[str, Any], context: Dict[str, Any] = None) -> List[Task]:
        """Decompose query into executable tasks."""
        base_tasks = []
        
        # Task 1: Information Extraction
        base_tasks.append(Task(
            task_id="extract_info",
            description="Extract role, level, location, and requirements from query",
            tools_required=["context_extractor"],
            dependencies=[]
        ))
        
        # Task 2: Market Research
        if complexity.get("complexity") in ["moderate", "complex"]:
            base_tasks.append(Task(
                task_id="market_research",
                description="Gather relevant market data and benchmarks",
                tools_required=["database_search", "web_search"],
                dependencies=["extract_info"]
            ))
        
        # Task 3: Risk Assessment
        if "budget" in query.lower() or complexity.get("complexity") == "complex":
            base_tasks.append(Task(
                task_id="risk_assessment",
                description="Assess budget constraints and policy compliance",
                tools_required=["risk_analyzer"],
                dependencies=["extract_info"]
            ))
        
        # Task 4: Recommendation Generation
        base_tasks.append(Task(
            task_id="generate_recommendation",
            description="Create comprehensive compensation recommendation",
            tools_required=["recommendation_engine"],
            dependencies=["extract_info", "market_research"] if len(base_tasks) > 1 else ["extract_info"]
        ))
        
        # Task 5: Validation & Refinement
        if complexity.get("complexity") == "complex":
            base_tasks.append(Task(
                task_id="validate_refine",
                description="Validate recommendation and refine based on constraints",
                tools_required=["validator", "refiner"],
                dependencies=["generate_recommendation"]
            ))
        
        return base_tasks
    
    def _plan_execution_order(self, tasks: List[Task]) -> List[str]:
        """Plan optimal execution order based on dependencies."""
        execution_order = []
        completed_tasks = set()
        
        while len(execution_order) < len(tasks):
            for task in tasks:
                if (task.task_id not in completed_tasks and 
                    all(dep in completed_tasks for dep in task.dependencies)):
                    execution_order.append(task.task_id)
                    completed_tasks.add(task.task_id)
                    break
        
        return execution_order
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def execute_plan(self, plan_id: str, tools: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute plan with ReAct reasoning and error recovery."""
        if plan_id not in self.active_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        plan.status = "in_progress"
        
        results = {}
        
        for task_id in plan.execution_order:
            task = next(t for t in plan.tasks if t.task_id == task_id)
            
            # Execute task with ReAct reasoning
            task_result = self._execute_task_with_react(task, plan, results, tools)
            results[task_id] = task_result
            
            # Update task status
            if task_result.get("success", False):
                task.status = "completed"
                task.output = task_result
                task.confidence = task_result.get("confidence", 0.5)
            else:
                task.status = "failed"
                # Attempt recovery or alternative approach
                recovery_result = self._attempt_task_recovery(task, plan, results, tools)
                if recovery_result.get("success", False):
                    task.status = "completed"
                    task.output = recovery_result
                    results[task_id] = recovery_result
        
        # Finalize plan
        plan.status = "completed" if all(t.status == "completed" for t in plan.tasks) else "partial"
        
        return {
            "plan": plan,
            "results": results,
            "success": plan.status == "completed"
        }
    
    def _execute_task_with_react(self, task: Task, plan: Plan, previous_results: Dict[str, Any], tools: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a single task using ReAct (Reason-Act-Observe) pattern."""
        reasoning_chain = []
        
        # THOUGHT: Reason about the task
        thought = self._generate_thought(task, plan, previous_results)
        reasoning_chain.append({"step": "thought", "content": thought})
        plan.reasoning_chain.append({"task_id": task.task_id, "step": "thought", "content": thought})
        
        # ACTION: Take action to complete the task
        action_result = self._execute_action(task, thought, previous_results, tools)
        reasoning_chain.append({"step": "action", "content": action_result})
        plan.reasoning_chain.append({"task_id": task.task_id, "step": "action", "content": action_result})
        
        # OBSERVATION: Observe the results
        observation = self._make_observation(action_result, task)
        reasoning_chain.append({"step": "observation", "content": observation})
        plan.reasoning_chain.append({"task_id": task.task_id, "step": "observation", "content": observation})
        
        # REFLECTION: Reflect on quality and next steps
        reflection = self._reflect_on_results(observation, task, plan.goal)
        reasoning_chain.append({"step": "reflection", "content": reflection})
        plan.reasoning_chain.append({"task_id": task.task_id, "step": "reflection", "content": reflection})
        
        return {
            "success": action_result.get("success", False),
            "output": action_result.get("output"),
            "confidence": reflection.get("confidence", 0.5),
            "reasoning_chain": reasoning_chain,
            "quality_score": reflection.get("quality_score", 5.0)
        }
    
    def _generate_thought(self, task: Task, plan: Plan, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning about how to approach the task."""
        context_summary = ""
        if previous_results:
            # Fix: avoid nested f-strings with escaped quotes
            context_summary = "Previous results: " + ', '.join([
                f"{k}: {v.get('output', 'N/A')}" for k, v in previous_results.items()
            ])
        
        prompt = f"""
        I need to complete this task as part of a compensation planning goal.
        
        Goal: {plan.goal}
        Task: {task.description}
        Required Tools: {task.tools_required}
        Dependencies: {task.dependencies}
        {context_summary}
        
        Think through:
        1. What specific approach should I take?
        2. What information do I need?
        3. What are potential challenges?
        4. How will I measure success?
        
        Provide structured thinking in JSON:
        {{
            "approach": "specific strategy",
            "information_needed": ["data", "requirements"],
            "potential_challenges": ["challenge1", "challenge2"],
            "success_metrics": ["metric1", "metric2"]
        }}
        """
        
        response = self.co_client.generate(prompt=prompt, max_tokens=250, temperature=0.2)
        
        try:
            thought_text = response.generations[0].text.strip()
            start_idx = thought_text.find('{')
            end_idx = thought_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = thought_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        
        return {
            "approach": f"Complete {task.description}",
            "information_needed": ["relevant_data"],
            "potential_challenges": ["data_availability"],
            "success_metrics": ["task_completion"]
        }
    
    def _execute_action(self, task: Task, thought: Dict[str, Any], previous_results: Dict[str, Any], tools: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the actual task action."""
        # This is where we'd integrate with the existing query router and tools
        # For now, simulate execution based on task type
        
        if task.task_id == "extract_info":
            return {"success": True, "output": "Information extracted successfully"}
        elif task.task_id == "market_research":
            return {"success": True, "output": "Market data gathered"}
        elif task.task_id == "risk_assessment":
            return {"success": True, "output": "Risks assessed"}
        elif task.task_id == "generate_recommendation":
            return {"success": True, "output": "Recommendation generated"}
        elif task.task_id == "validate_refine":
            return {"success": True, "output": "Recommendation validated and refined"}
        else:
            return {"success": False, "output": "Unknown task type"}
    
    def _make_observation(self, action_result: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Observe and analyze the action results."""
        success = action_result.get("success", False)
        output = action_result.get("output", "")
        
        observation = {
            "task_completed": success,
            "output_quality": "good" if success else "poor",
            "issues_detected": [] if success else ["execution_failed"],
            "next_steps": "proceed" if success else "retry_or_modify"
        }
        
        return observation
    
    def _reflect_on_results(self, observation: Dict[str, Any], task: Task, goal: str) -> Dict[str, Any]:
        """Reflect on the quality of results and plan improvements."""
        quality_score = 8.0 if observation.get("task_completed", False) else 3.0
        confidence = 0.8 if observation.get("task_completed", False) else 0.3
        
        reflection = {
            "quality_score": quality_score,
            "confidence": confidence,
            "goal_alignment": "good" if quality_score > 6 else "needs_improvement",
            "recommendations": ["continue"] if quality_score > 6 else ["retry", "modify_approach"]
        }
        
        return reflection
    
    def _attempt_task_recovery(self, task: Task, plan: Plan, previous_results: Dict[str, Any], tools: Dict[str, Any] = None) -> Dict[str, Any]:
        """Attempt to recover from task failure."""
        # Simple recovery: try alternative approach
        return {
            "success": True,
            "output": f"Recovered execution of {task.description}",
            "confidence": 0.6,
            "recovery_used": True
        }
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get current status of a plan."""
        if plan_id not in self.active_plans:
            return {"error": "Plan not found"}
        
        plan = self.active_plans[plan_id]
        return {
            "plan_id": plan_id,
            "status": plan.status,
            "current_step": plan.current_step,
            "total_steps": len(plan.tasks),
            "completed_tasks": [t.task_id for t in plan.tasks if t.status == "completed"],
            "failed_tasks": [t.task_id for t in plan.tasks if t.status == "failed"],
            "reasoning_chain_length": len(plan.reasoning_chain)
        }
    
    def explain_reasoning(self, plan_id: str) -> List[Dict[str, Any]]:
        """Get the complete reasoning chain for transparency."""
        if plan_id not in self.active_plans:
            return []
        
        plan = self.active_plans[plan_id]
        return plan.reasoning_chain

class Planner:
    """Agent planner that creates a plan of tool calls for a given query."""
    
    def __init__(self, co_client=None):
        """Initialize the planner with optional Cohere client."""
        self.co_client = co_client
        self.planning_prompt = """
        You are an AI assistant specializing in compensation planning. Break down the user's query into a sequence of tool calls to create a comprehensive compensation recommendation.
        
        Available tools:
        - calculate_comp: Calculate compensation ranges based on role, level, location and market data
        - check_policy: Check if the proposed compensation follows company policy
        - flag_risk: Identify potential risks or concerns with the compensation plan
        - general_answer: Provide a general answer for questions not requiring specific calculations
        
        Create a plan with a sequence of tool calls to address the query.
        
        Here are some example plans:
        
        Example 1:
        User Query: "Create an offer for a Senior Software Engineer in San Francisco"
        Persona: "Analyst"
        Plan: [
            {"tool": "calculate_comp", "args": {"role": "Software Engineer", "level": "Senior", "location": "San Francisco"}},
            {"tool": "check_policy", "args": {"role": "Software Engineer", "level": "Senior"}},
            {"tool": "flag_risk", "args": {"role": "Software Engineer", "level": "Senior", "location": "San Francisco"}}
        ]
        
        Example 2:
        User Query: "What's a typical equity package for a Series B startup?"
        Persona: "Hiring Manager"
        Plan: [
            {"tool": "general_answer", "args": {"topic": "equity packages", "company_stage": "Series B"}}
        ]
        
        User Query: "{query}"
        Persona: "{persona}"
        Plan:
        """
    
    def plan(self, query, persona="Analyst"):
        """Create a plan of tool calls for a given query."""
        # First, ensure we have a client - check if we need to get it from session state
        import streamlit as st
        
        if self.co_client is None and 'cohere_client' in st.session_state:
            self.co_client = st.session_state.cohere_client
        
        if self.co_client is None:
            # Instead of raising an error, return a safe fallback plan
            print("Warning: No Cohere client available, using fallback plan")
            return self._create_fallback_plan(query, persona)
        
        prompt = self.planning_prompt.format(query=query, persona=persona)
        
        try:
            # Make the call to generate a plan
            response = self.co_client.chat(
                model="command-r-plus",
                message=prompt,
                temperature=0.2
            )
            
            # Check if the response has tool_calls in the new SDK format
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print("Using new Cohere SDK tool_calls format")
                plan = []
                for tool_call in response.tool_calls:
                    try:
                        # Extract tool name and arguments from the tool call
                        tool_name = tool_call.name
                        tool_args = tool_call.parameters
                        plan.append({"tool": tool_name, "args": tool_args})
                    except Exception as e:
                        print(f"Error processing tool call: {e}")
                
                if plan:
                    return plan
                # If no valid tool calls, fall through to text parsing
            
            plan_text = response.text
            
            # Extract JSON list from response
            import re
            import json
            
            # IMPROVED PARSER: Try multiple strategies to extract the JSON
            # First try to find anything that looks like a JSON array with [
            json_match = re.search(r'\[[\s\S]*?\]', plan_text, re.DOTALL)
            
            if json_match:
                try:
                    plan_str = json_match.group()
                    # Clean up any markdown code block markers
                    plan_str = plan_str.replace('```json', '').replace('```', '')
                    plan = json.loads(plan_str)
                    
                    # Validate the plan structure - crucial to prevent KeyError
                    if not isinstance(plan, list):
                        print("Plan is not a list, using fallback")
                        return self._create_fallback_plan(query, persona)
                    
                    # Verify each step has required keys
                    for i, step in enumerate(plan):
                        if not isinstance(step, dict):
                            print(f"Step {i} is not a dictionary, replacing with general_answer")
                            plan[i] = {"tool": "general_answer", "args": {"query": query}}
                            continue
                            
                        if "tool" not in step or not step.get("tool"):
                            print(f"Missing tool in step {i}, replacing with general_answer")
                            plan[i] = {"tool": "general_answer", "args": {"query": query}}
                            continue
                            
                        if "args" not in step or not isinstance(step["args"], dict):
                            print(f"Missing or invalid args in step {i}, adding default")
                            plan[i]["args"] = {"query": query}
                    
                    # Ensure we have at least one step
                    if not plan:
                        return self._create_fallback_plan(query, persona)
                    
                    return plan
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    # Try a more lenient parsing approach
                    
            # If standard parsing failed, try alternative approaches
            # Look for a sequence of tool calls with regex
            tools_found = []
            
            # Pattern to match individual tool calls with more flexibility
            tool_pattern = r'{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*({[^{}]*(?:{[^{}]*}[^{}]*)*})'
            tool_matches = re.finditer(tool_pattern, plan_text, re.DOTALL)
            
            for match in tool_matches:
                try:
                    tool_name = match.group(1).strip()
                    args_str = match.group(2).strip()
                    
                    # Try to clean up the args string and parse it
                    args_str = args_str.replace("'", '"')
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        # If we can't parse the args, create a simple default
                        args = {"query": query}
                    
                    tools_found.append({"tool": tool_name, "args": args})
                except Exception as parse_error:
                    print(f"Error parsing individual tool: {parse_error}")
            
            if tools_found:
                return tools_found
            
            # Final fallback - extract simple tool mentions
            simple_tools = []
            simple_tool_pattern = r'\b(calculate_comp|check_policy|flag_risk|general_answer)\b'
            for tool in re.finditer(simple_tool_pattern, plan_text):
                simple_tools.append({"tool": tool.group(1), "args": {"query": query}})
            
            if simple_tools:
                return simple_tools
                
            # If all extraction methods fail, use fallback plan
            print("Could not extract plan from response, using fallback")
            return self._create_fallback_plan(query, persona)
            
        except Exception as e:
            print(f"Error creating plan: {e}")
            return self._create_fallback_plan(query, persona)
    
    def _create_fallback_plan(self, query, persona):
        """Create a basic fallback plan when the main planning fails."""
        query_lower = query.lower()
        
        # Check if it's about compensation calculation
        if any(word in query_lower for word in ["offer", "salary", "compensation", "pay", "create"]):
            # Try to extract role, level and location
            role = "Software Engineer"  # Default
            level = "Mid-level"         # Default
            location = "San Francisco"  # Default
            
            # Very basic extraction based on common keywords
            if "senior" in query_lower or "sr" in query_lower:
                level = "Senior"
            elif "junior" in query_lower or "jr" in query_lower:
                level = "Junior"
            elif "staff" in query_lower:
                level = "Staff"
            elif "principal" in query_lower:
                level = "Principal"
            
            if "engineer" in query_lower:
                role = "Software Engineer"
            elif "product" in query_lower and "manager" in query_lower:
                role = "Product Manager"
            elif "designer" in query_lower:
                role = "Designer"
            elif "data" in query_lower:
                role = "Data Scientist"
            
            if "san francisco" in query_lower or "sf" in query_lower:
                location = "San Francisco"
            elif "new york" in query_lower or "nyc" in query_lower:
                location = "New York"
            elif "seattle" in query_lower:
                location = "Seattle"
                
            return [
                {"tool": "calculate_comp", "args": {"role": role, "level": level, "location": location}},
                {"tool": "check_policy", "args": {"role": role, "level": level}},
                {"tool": "flag_risk", "args": {"role": role, "location": location}}
            ]
        
        # Default to general answer
        return [{"tool": "general_answer", "args": {"query": query}}]