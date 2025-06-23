"""
Streamlit Community Cloud Entry Point
This file serves as the main entry point for deploying to Streamlit Community Cloud.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the main Compensation Planner app
from pages.Compensation_Planner import main

if __name__ == "__main__":
    main() 