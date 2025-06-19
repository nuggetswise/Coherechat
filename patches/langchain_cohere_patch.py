"""
Patch for langchain-cohere to address Pydantic v2 compatibility issues
The langchain-cohere package is using __modify_schema__ which is not supported in Pydantic v2.
"""
import sys
from types import ModuleType
from functools import wraps
import importlib

def patch_langchain_cohere():
    """
    Apply patch to fix Pydantic v2 compatibility issues in langchain-cohere.
    This fixes the PydanticUserError related to __modify_schema__ method.
    """
    try:
        # First check if langchain_cohere is already imported
        if "langchain_cohere" in sys.modules:
            print("⚠️ langchain_cohere is already imported, patch may not be fully effective")
            
        # Create a patched version of the module
        patched_module = create_patched_module()
        
        # Replace the module in sys.modules
        if patched_module:
            sys.modules["langchain_cohere"] = patched_module
            print("✅ Successfully patched langchain_cohere for Pydantic v2 compatibility")
            return True
        else:
            print("❌ Failed to create patched module")
            return False
            
    except Exception as e:
        print(f"❌ Error patching langchain_cohere: {str(e)}")
        return False

def create_patched_module():
    """Create a patched version of the langchain_cohere module"""
    try:
        # Import the original module
        original_module = importlib.import_module("langchain_cohere")
        
        # Create a new module to replace it
        patched_module = ModuleType("langchain_cohere")
        patched_module.__dict__.update(original_module.__dict__)
        
        # Patch the ChatCohere class if it exists
        if hasattr(original_module, "chat_models") and hasattr(original_module.chat_models, "ChatCohere"):
            # Create a modified import function for chat_models
            original_import = __import__
            
            @wraps(original_import)
            def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
                """Patched import function to modify the ChatCohere class"""
                module = original_import(name, globals, locals, fromlist, level)
                
                if name == "langchain_cohere.chat_models" or name == "langchain_cohere.llms":
                    # Apply the patch to fix SecretStr schema issue
                    try:
                        # Fix BaseCohere class
                        if hasattr(module, "BaseCohere"):
                            # Add __get_pydantic_json_schema__ method to SecretStr fields
                            if hasattr(module.BaseCohere, "__annotations__"):
                                for field_name, field_type in module.BaseCohere.__annotations__.items():
                                    if "SecretStr" in str(field_type):
                                        # We need to monkey patch this at runtime
                                        from pydantic import SecretStr
                                        if not hasattr(SecretStr, "__get_pydantic_json_schema__"):
                                            def get_pydantic_json_schema(cls, _schema, field_schema):
                                                field_schema["type"] = "string"
                                                field_schema["format"] = "password"
                                                return field_schema
                                            
                                            SecretStr.__get_pydantic_json_schema__ = classmethod(get_pydantic_json_schema)
                                            print(f"✅ Added __get_pydantic_json_schema__ to SecretStr")
                    except Exception as e:
                        print(f"⚠️ Error patching BaseCohere: {str(e)}")
                
                return module
            
            # Replace the built-in __import__ function when importing langchain_cohere submodules
            builtins_module = sys.modules["builtins"]
            original_builtins_import = builtins_module.__import__
            builtins_module.__import__ = patched_import
            
            # Force reimport of chat_models and llms to apply the patch
            try:
                if "langchain_cohere.chat_models" in sys.modules:
                    del sys.modules["langchain_cohere.chat_models"]
                if "langchain_cohere.llms" in sys.modules:
                    del sys.modules["langchain_cohere.llms"]
                    
                # Reimport them under the patched module
                patched_module.chat_models = importlib.import_module("langchain_cohere.chat_models")
                patched_module.llms = importlib.import_module("langchain_cohere.llms")
                
                # Restore original import
                builtins_module.__import__ = original_builtins_import
            except Exception as e:
                # Restore original import if there was an error
                builtins_module.__import__ = original_builtins_import
                print(f"❌ Error reimporting modules: {str(e)}")
                return None
        
        # Update other attributes
        for attr_name in dir(original_module):
            if attr_name.startswith("__") or attr_name in ["chat_models", "llms"]:
                continue
                
            attr = getattr(original_module, attr_name)
            setattr(patched_module, attr_name, attr)
        
        return patched_module
        
    except Exception as e:
        print(f"❌ Error creating patched module: {str(e)}")
        return None

# Alternative approach: direct monkey patching of SecretStr
def patch_pydantic_secret_str():
    """
    Directly patch the Pydantic SecretStr class to add __get_pydantic_json_schema__.
    This is a more targeted approach that modifies Pydantic itself.
    """
    try:
        from pydantic import SecretStr
        
        # Check if patch is needed
        if hasattr(SecretStr, "__get_pydantic_json_schema__"):
            print("✅ SecretStr already has __get_pydantic_json_schema__")
            return True
            
        # Add the method
        def get_pydantic_json_schema(cls, _schema, field_schema):
            field_schema["type"] = "string"
            field_schema["format"] = "password"
            return field_schema
        
        SecretStr.__get_pydantic_json_schema__ = classmethod(get_pydantic_json_schema)
        print("✅ Successfully added __get_pydantic_json_schema__ to SecretStr")
        return True
        
    except Exception as e:
        print(f"❌ Error patching SecretStr: {str(e)}")
        return False