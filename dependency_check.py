"""
Dependency Check Module
Validates that all required dependencies are available and properly configured.
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple, Any

# Define required dependencies with version constraints
REQUIRED_DEPENDENCIES = {
    "streamlit": ">=1.36.0",
    "cohere": ">=5.15.0", 
    "openai": ">=1.38.0",
    "langchain": ">=0.1.13",
    "langchain_core": ">=0.1.42",
    "pandas": ">=2.2.3",
    "numpy": ">=1.26.3",
    "pydantic": ">=1.10.13",
    "requests": ">=2.32.4",
    "dotenv": ">=1.0.0"
}

# Optional dependencies that enhance functionality
OPTIONAL_DEPENDENCIES = {
    "tiktoken": ">=0.5.0",
    "pypdf": ">=4.3.1",
    "docx": ">=1.1.2",
    "bs4": ">=4.12.3",
    "streamlit_option_menu": ">=0.4.0"
}

def check_sqlite_version() -> Tuple[bool, str]:
    """Check if SQLite version meets requirements"""
    try:
        import sqlite3
        version = sqlite3.sqlite_version
        version_tuple = tuple(map(int, version.split('.')))
        
        # Check if version is at least 3.35.0
        if version_tuple >= (3, 35, 0):
            return True, f"SQLite version {version} meets requirements"
        else:
            return False, f"SQLite version {version} is too old (requires >= 3.35.0)"
    except Exception as e:
        return False, f"Error checking SQLite version: {str(e)}"

def check_dependency(name: str, version_constraint: str) -> Tuple[bool, str, str]:
    """Check if a dependency is available and meets version requirements"""
    try:
        # Try to import the module
        module = importlib.import_module(name)
        
        # Get version if available
        version = getattr(module, '__version__', 'unknown')
        
        # For now, just check if it's importable
        # In a production system, you'd want to use packaging.version to check constraints
        return True, version, f"✅ {name} {version} is available"
        
    except ImportError as e:
        return False, "not installed", f"❌ {name} is not installed: {str(e)}"
    except Exception as e:
        # Special handling for packages that can raise RuntimeError
        if name == "chromadb":
            return False, "error", f"❌ {name} import failed: {str(e)}"
        else:
            return False, "error", f"❌ {name} import failed: {str(e)}"

def run_dependency_check() -> Dict[str, Any]:
    """Run comprehensive dependency check"""
    results = {
        "required": {},
        "optional": {},
        "sqlite": {},
        "overall_status": "unknown"
    }
    
    # Check required dependencies
    print("🔍 Checking required dependencies...")
    for dep_name, version_constraint in REQUIRED_DEPENDENCIES.items():
        available, version, message = check_dependency(dep_name, version_constraint)
        results["required"][dep_name] = {
            "available": available,
            "version": version,
            "message": message,
            "required_version": version_constraint
        }
        print(f"  {message}")
    
    # Check optional dependencies
    print("\n🔍 Checking optional dependencies...")
    for dep_name, version_constraint in OPTIONAL_DEPENDENCIES.items():
        available, version, message = check_dependency(dep_name, version_constraint)
        results["optional"][dep_name] = {
            "available": available,
            "version": version,
            "message": message,
            "required_version": version_constraint
        }
        print(f"  {message}")
    
    # Check SQLite version
    print("\n🔍 Checking SQLite version...")
    sqlite_ok, sqlite_message = check_sqlite_version()
    results["sqlite"] = {
        "available": sqlite_ok,
        "message": sqlite_message
    }
    print(f"  {sqlite_message}")
    
    # Determine overall status
    required_failures = [dep for dep, info in results["required"].items() if not info["available"]]
    optional_failures = [dep for dep, info in results["optional"].items() if not info["available"]]
    
    if required_failures:
        results["overall_status"] = "failed"
        print(f"\n❌ Dependency check failed. Missing required dependencies: {', '.join(required_failures)}")
        print("   Please install missing dependencies: pip install -r requirements-minimal.txt")
    else:
        results["overall_status"] = "passed"
        print(f"\n✅ All required dependencies are available!")
        
        # Provide guidance for missing optional dependencies
        if optional_failures:
            print(f"\n⚠️  Some optional dependencies are missing: {', '.join(optional_failures)}")
            if "tiktoken" in optional_failures:
                print("   Note: tiktoken build issues are common on Python 3.13+ and Linux systems")
                print("   The app works perfectly without tiktoken. If you need it, try:")
                print("   - pip install tiktoken --no-build-isolation")
                print("   - or use Python 3.11/3.12 instead of 3.13+")
            print("   These are optional and won't affect core functionality.")
    
    return results

def display_dependency_status(results: Dict[str, Any]):
    """Display dependency status in a user-friendly format"""
    import streamlit as st
    
    st.markdown("## 🔍 Dependency Status")
    
    # Required dependencies
    st.markdown("### Required Dependencies")
    for dep_name, info in results["required"].items():
        if info["available"]:
            st.success(f"✅ {dep_name} {info['version']}")
        else:
            st.error(f"❌ {dep_name}: {info['message']}")
    
    # Optional dependencies
    st.markdown("### Optional Dependencies")
    for dep_name, info in results["optional"].items():
        if info["available"]:
            st.success(f"✅ {dep_name} {info['version']}")
        else:
            st.warning(f"⚠️ {dep_name}: {info['message']}")
    
    # SQLite status
    st.markdown("### System Dependencies")
    if results["sqlite"]["available"]:
        st.success(f"✅ SQLite: {results['sqlite']['message']}")
    else:
        st.warning(f"⚠️ SQLite: {results['sqlite']['message']}")
    
    # Overall status
    if results["overall_status"] == "passed":
        st.success("🎉 All required dependencies are available!")
        
        # Show guidance for missing optional dependencies
        optional_failures = [dep for dep, info in results["optional"].items() if not info["available"]]
        if optional_failures:
            st.warning(f"⚠️ Some optional dependencies are missing: {', '.join(optional_failures)}")
            if "tiktoken" in optional_failures:
                st.info("💡 **tiktoken** build issues are common on Python 3.13+ and Linux systems. The app works perfectly without it.")
    else:
        st.error("❌ Some required dependencies are missing. Please install them.")

if __name__ == "__main__":
    results = run_dependency_check()
    print(f"\nOverall status: {results['overall_status']}")