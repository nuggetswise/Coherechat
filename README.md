# Compensation Planner - AI-Powered Agent Workflow

## Overview
This application is an AI-powered multi-agent system for creating and evaluating compensation packages. It employs a collaborative workflow of specialized AI agents (Recruitment Manager, HR Director, and Hiring Manager) to analyze requirements, ensure policy compliance, and make final compensation decisions with transparent evaluation.

- **Audience:**
  - **Developers:** Learn about the architecture, agent workflow, and tech stack for extending or maintaining the system.
  - **HR Professionals:** Understand how the system works, the agent decision process, and how AI is used to assist compensation planning.

---

## End-to-End Flow (Non-Technical)

1. **You provide job requirements** (e.g., "Create a compensation package for a Senior Software Engineer in San Francisco") in the input interface.
2. **The Recruitment Manager agent** analyzes your requirements and creates a detailed compensation package based on market data and best practices.
3. **The HR Director agent** reviews the package for policy compliance, internal equity, and alignment with company guidelines.
4. **The Hiring Manager agent** makes the final decision, considering the package details and HR feedback.
5. **The system evaluates each agent's output** for quality, relevance, and accuracy, providing transparent metrics.
6. **You receive a complete compensation package** with detailed analysis and evaluation.

---

## End-to-End Flow (Technical)

1. **User Input:**
   - User submits requirements via the Streamlit UI (`pages/Compensation_Planner.py`).

2. **Agent Workflow Pipeline:**
   - **Recruitment Manager:**
     - Processes user query using Cohere's LLM.
     - Generates a detailed compensation package with salary, bonus, equity, and benefits.
     - Extracts role, level, and location information using regex pattern matching.
   - **HR Director:**
     - Reviews the compensation package against policy guidelines.
     - Provides feedback on compliance and internal equity.
     - Assigns a confidence score to the assessment.
   - **Hiring Manager:**
     - Makes final approval decision based on package details and HR feedback.
     - Provides comments on budget considerations, equity concerns, and any risk flags.
     - Always approves but may include improvement recommendations.

3. **Evaluation Framework:**
   - Each agent's output is evaluated by a separate evaluation process.
   - Scores are generated for relevance, factual accuracy, and groundedness.
   - Overall quality score and specific feedback are provided.

4. **UI & Transparency:**
   - Streamlit UI displays each agent's output in a sequential workflow.
   - Progress indicators show the current stage of the process.
   - Quality evaluations are displayed for each agent with detailed metrics.

---

## Tech Stack
- **Frontend/UI:** Streamlit
- **LLM:** Cohere (primary), OpenAI (fallback)
- **Data Processing:** Pandas for CSV handling
- **Text Processing:** Regular expressions for structured data extraction
- **Evaluation:** LLM-based, using custom evaluation prompts
- **Configuration & Prompts:** Centralized in `comp_planner/persona_prompts.py`
- **Database Compatibility:** CSV for straightforward data management
- **Schema Validation:** Custom validation logic in `comp_planner/schema_validation.py`
- **Patches:** Custom patches for Cohere types via `patches/cohere_types_patch.py`

---

## Key Files & Structure
- `pages/Compensation_Planner.py` — Main app logic and UI
- `agents/offer_chain.py` — Core agent workflow implementation
- `comp_planner/persona_prompts.py` — All prompt templates for agents
- `comp_planner/schema_validation.py` — Validation logic for agent outputs
- `comp_planner/evaluation_framework.py` — Custom evaluation logic
- `data/Compensation Data.csv` — Reference compensation database
- `patches/` — Contains patches for third-party libraries to enhance functionality

---

## Agent System
- **Recruitment Manager:**
  - Creates detailed compensation packages with specific values for base salary, bonus, equity, and benefits.
  - Extracts and infers role, level, and location from the user query.
  - Uses a combination of market data and industry best practices.

- **HR Director:**
  - Reviews packages for policy compliance and internal equity.
  - Provides a confidence score on the assessment.
  - Suggests changes when necessary.

- **Hiring Manager:**
  - Makes the final decision on the compensation package.
  - Identifies any equity concerns or budget constraints.
  - Provides detailed feedback on the package.

- **Evaluation System:**
  - Evaluates each agent's output based on relevance, factual accuracy, and groundedness.
  - Provides an overall quality score.
  - Identifies strengths and areas for improvement.

---

## How to Run Locally
1. Clone the repo and ensure `data/Compensation Data.csv` is present.
2. Install requirements: `pip install -r requirements.txt`
3. Set your Cohere and OpenAI API keys in your environment or Streamlit secrets.
4. Run: `streamlit run pages/Compensation_Planner.py --server.port 8505`
5. Open [http://localhost:8505](http://localhost:8505) in your browser.

---

## Path Resolution
The system uses a robust path resolution mechanism to find the compensation database:
```python
# Get the absolute path to the current file (Compensation_Planner.py)
current_file = os.path.abspath(__file__)
# Get the directory containing the current file (pages/)
current_dir = os.path.dirname(current_file)
# Get the project root directory (one level up from pages/)
project_root = os.path.dirname(current_dir)
# Construct the absolute path to the compensation data file
data_path = os.path.join(project_root, "data", "Compensation Data.csv")
```
This ensures the app works in both local and production environments.

---

## How to Deploy
- Make sure `data/Compensation Data.csv` is included in your deployment.
- Deploy to Streamlit Community Cloud or your own server.
- The app will use the internal DB for all agent operations.
- Set the required API keys as environment variables or in Streamlit secrets.

---

## FAQ
- **Why are all Hiring Manager decisions "Approved"?**
  - The system is designed to always approve packages but provides feedback for improvements.
- **Are evaluation scores hardcoded?**
  - No, all evaluation scores and feedback are generated by the LLM in real time.
- **Can I add more data?**
  - Yes, update `Compensation Data.csv` and restart the app to incorporate new market data.
- **How do I customize agent personalities?**
  - Edit the prompt templates in `comp_planner/persona_prompts.py` to adjust agent behaviors.
- **Can I use a different LLM provider?**
  - Yes, the system is designed to work with both Cohere and OpenAI, and can be extended to other providers.

---

## Contact & Support
For questions or contributions, please contact the development team or open an issue in this repository.

---

## 🧠 System Logic, Agents, and Configuration

### What Happens When You Submit Requirements?
1. **Your requirements are processed by the Recruitment Manager agent** to generate a detailed compensation package.
2. **The HR Director agent reviews the package** for policy compliance and internal equity.
3. **The Hiring Manager agent makes the final decision** based on the package and HR feedback.
4. **Each agent's output is evaluated** for quality, relevance, and accuracy.
5. **You receive a complete compensation package** with transparent evaluation metrics.

---

### 👔 How Agent Workflow Works
- **Recruitment Manager:**
  - Analyzes your requirements to understand the role, level, and location.
  - Creates a comprehensive compensation package with specific values.
  - Extracts structured data like role and level using pattern matching.
- **HR Director:**
  - Reviews the package against company policies and industry standards.
  - Provides a confidence score (1-10) on the assessment.
  - Identifies any policy violations or equity concerns.
- **Hiring Manager:**
  - Reviews both the package and HR feedback.
  - Makes a final decision (always approved, but with feedback).
  - Identifies any risk flags or budget constraints.

---

### 📊 How AI Evaluation Works
- **After each agent completes its task,** the system evaluates the output on:
  - **Relevance:** How well did the output address the requirements?
  - **Factual Accuracy:** Is the information correct and supported?
  - **Groundedness:** Does the output avoid speculation?
- **Each dimension receives a score (1-10) and detailed feedback.**
- **No scores or feedback are hardcoded**—everything is generated live for each case.
- **You see a transparent evaluation table with full feedback available.**

---

### ⚙️ What Can Be Tweaked (Configuration)
- **All agent prompts, evaluation criteria, and workflows are configurable.**
- **You can adjust:**
  - The personality and focus of each agent through prompt engineering
  - The evaluation criteria and scoring weights
  - The detail level of the compensation packages
  - The sources of market data used for reference
- **These settings are typically managed in the persona_prompts.py file.**

---

### 🛡️ Privacy & Data Handling
- **Your requirements and generated packages are only used for your session.**
- **The internal compensation database is never exposed directly—only processed insights are shown.**
- **All evaluations are performed within the system and not shared externally.**

---

### 📝 Example Agent Prompts
- **Recruitment Manager Prompt:**
  - "You are an expert Compensation & Benefits Manager. Create a detailed compensation package based on the user's requirements, market data, and best practices..."
- **HR Director Prompt:**
  - "You are an HR Director reviewing a compensation package. Evaluate the package for policy compliance, internal equity, and alignment with company guidelines..."
- **Hiring Manager Prompt:**
  - "You are a Hiring Manager making the final decision on a compensation package. Review the package and HR feedback to determine approval status..."

---

### 💡 Summary for All Users
- **You provide requirements for a compensation package.**
- **Three specialized AI agents collaborate to create, review, and approve the package.**
- **Each agent's work is evaluated for quality and transparency.**
- **You receive a comprehensive, well-justified compensation package.**
- **All agent behaviors and evaluation criteria can be customized to fit your needs.**


