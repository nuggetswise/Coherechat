# Python 3.13+ Installation Guide

## Overview
If you're using Python 3.13+ and encountering `tiktoken` build issues, this guide will help you get the Compensation Planner app running.

## The Issue
`tiktoken` has known build issues on Python 3.13+ and some Linux systems. This is a common problem and doesn't affect the core functionality of the app.

## Quick Fix

### Option 1: Use the Minimal Requirements (Recommended)
```bash
# Install without tiktoken
pip install -r requirements-minimal.txt

# Run the app
streamlit run pages/Compensation_Planner.py
```

### Option 2: Try Alternative tiktoken Installation
If you really need tiktoken for token counting:

```bash
# Method 1: No build isolation
pip install tiktoken --no-build-isolation

# Method 2: Binary only
pip install tiktoken --only-binary=all

# Method 3: Specific version
pip install tiktoken==0.5.0
```

### Option 3: Use Python 3.11 or 3.12
If you need tiktoken and the above methods don't work:

```bash
# Create a new virtual environment with Python 3.11 or 3.12
python3.11 -m venv .venv
# or
python3.12 -m venv .venv

source .venv/bin/activate
pip install -r requirements.txt
```

## What You'll See Without tiktoken

The app will work perfectly, but you may see:
- Warnings about token counting not being available
- Some features may show "token count unavailable"

**This does NOT affect:**
- ✅ Core compensation planning functionality
- ✅ Multi-agent workflow
- ✅ RAG system
- ✅ Evaluation framework
- ✅ Market data analysis

## Verification

After installation, run:
```bash
python dependency_check.py
```

You should see:
```
✅ All required dependencies are available!
⚠️ Some optional dependencies are missing (this is normal)
```

## Troubleshooting

### Still Having Issues?
1. **Check Python version**: `python --version`
2. **Try minimal requirements**: `pip install -r requirements-minimal.txt`
3. **Skip tiktoken entirely**: The app works fine without it
4. **Use Python 3.11/3.12**: These versions have better tiktoken compatibility

### Common Error Messages
```
× Failed to download and build `tiktoken==0.7.0`
```
**Solution**: Use `requirements-minimal.txt` instead of `requirements.txt`

```
Build backend failed to build wheel
```
**Solution**: Try `pip install tiktoken --no-build-isolation`

## Support

If you continue to have issues:
1. The app works perfectly without tiktoken
2. Consider using Python 3.11 or 3.12 for full compatibility
3. Check the main README.md for general troubleshooting 