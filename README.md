# Compensation Planner - AI-Powered Agent Workflow

## Overview
This application is an AI-powered system for creating and evaluating compensation packages. It employs a collaborative workflow of specialized AI agents to analyze requirements, ensure policy compliance, and make final compensation decisions with transparent evaluation.

- **Audience:**
  - **Developers:** Learn about the architecture, agent workflow, and tech stack for extending or maintaining the system.
  - **HR Professionals:** Understand how the system works, the agent decision process, and how AI is used to assist compensation planning.

---

## End-to-End Flow (Non-Technical)

1. **You provide job requirements** (e.g., "Create a compensation package for a Senior Software Engineer in San Francisco")
2. **Recruitment Manager Agent** analyzes market data and creates initial compensation recommendations
3. **HR Director Agent** reviews the recommendations against company policies and compliance requirements
4. **Hiring Manager Agent** makes final approval decisions considering business impact and budget constraints
5. **You receive** a comprehensive compensation package with detailed breakdown and justification

---

## Key Features

### 🤖 **Multi-Agent Workflow**
- **Recruitment Manager**: Market research and initial salary recommendations
- **HR Director**: Policy compliance and legal review
- **Hiring Manager**: Final approval and business case evaluation

### 📊 **Data-Driven Insights**
- Real-time market data analysis
- Salary benchmarking against industry standards
- Equity and bonus structure recommendations
- Total compensation calculations

### 🔍 **RAG (Retrieval-Augmented Generation)**
- Document upload and analysis
- Policy document processing
- Context-aware recommendations

### 📈 **Evaluation Framework**
- Confidence scoring for recommendations
- Agent decision transparency
- Performance metrics tracking

---

## How to Run Locally

1. **Clone the repo** and ensure `data/Compensation Data.csv` is present
2. **Install requirements**: `pip install -r requirements-minimal.txt`
3. **Set your API keys** in your environment or Streamlit secrets
4. **Run**: `streamlit run pages/Compensation_Planner.py --server.port 8505`
5. **Open** [http://localhost:8505](http://localhost:8505) in your browser

### Quick Start Script
```bash
# Use the provided installation script
./install-deps.sh

# Or manually install dependencies
pip install -r requirements-minimal.txt

# Run the app
streamlit run pages/Compensation_Planner.py
```

---

## API Keys Required

The application requires the following API keys:

### **Cohere API Key** (Required)
- Used for the main AI agent workflow
- Get your key at: https://console.cohere.ai/

### **OpenAI API Key** (Optional)
- Used for additional AI capabilities
- Get your key at: https://platform.openai.com/

### **Google AI API Key** (Optional)
- Used for Google's generative AI features
- Get your key at: https://makersuite.google.com/app/apikey

---

## Configuration

### Environment Variables
```bash
export COHERE_API_KEY="your-cohere-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_API_KEY="your-google-api-key"
```

### Streamlit Secrets (Recommended)
Create `.streamlit/secrets.toml`:
```toml
[cohere]
COHERE_API_KEY = "your-cohere-api-key"

[openai]
OPENAI_API_KEY = "your-openai-api-key"

[google]
GOOGLE_API_KEY = "your-google-api-key"
```

---

## Architecture

### **Core Components**
- `pages/Compensation_Planner.py` - Main application entry point
- `comp_planner/` - Core business logic and agent implementations
- `agents/` - Agent workflow definitions
- `data/` - Market data and reference materials

### **Agent Workflow**
1. **Input Processing**: Parse user requirements and context
2. **Market Analysis**: Research current market conditions
3. **Policy Review**: Validate against company policies
4. **Final Approval**: Business case evaluation and approval
5. **Output Generation**: Structured compensation recommendation

### **Data Flow**
```
User Input → Agent Workflow → Market Data → Policy Check → Approval → Output
```

---

## Dependencies

### **Core Dependencies**
- `streamlit==1.36.0` - Web application framework
- `cohere==5.15.0` - AI provider for agent workflow
- `openai==1.38.0` - Additional AI capabilities
- `langchain==0.1.13` - LLM framework
- `pandas==2.2.3` - Data processing
- `pydantic>=2.4.2` - Data validation

### **Optional Dependencies**
- `tiktoken>=0.5.0` - Token counting (may have build issues on some systems)
- `pypdf==4.3.1` - PDF document processing
- `python-docx==1.1.2` - Word document processing
- `beautifulsoup4==4.12.3` - Web scraping capabilities

---

## Troubleshooting

### **Common Issues**

#### **Dependency Installation Problems**
```bash
# If you encounter build issues with tiktoken
pip install -r requirements-minimal.txt

# The app will work without tiktoken, just with token counting warnings
```

#### **API Key Issues**
- Ensure your API keys are properly set in environment variables or Streamlit secrets
- Check that your API keys have sufficient credits/quota
- Verify the API keys are for the correct services (Cohere, OpenAI, Google)

#### **Port Already in Use**
```bash
# Use a different port
streamlit run pages/Compensation_Planner.py --server.port 8506
```

---

## Development

### **Adding New Agents**
1. Create agent class in `comp_planner/`
2. Implement required methods
3. Add to workflow in main application
4. Update UI components

### **Extending Data Sources**
1. Add new data files to `data/` directory
2. Update data loading logic
3. Modify agent prompts to use new data
4. Test with sample queries

### **Customizing Workflows**
1. Modify agent prompts in `comp_planner/`
2. Adjust workflow logic in main application
3. Update UI components for new features
4. Test thoroughly with various inputs

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the dependency requirements
3. Ensure your API keys are properly configured
4. Check that all required data files are present


