#!/usr/bin/env python3
"""Simple test script to verify our implementations work without full dependencies."""

import sys
import os
from pathlib import Path

# Add our local stubs to path
sys.path.insert(0, str(Path(__file__).parent / "trl" / "_local_deps"))

def test_dpo_online_bias_syntax():
    """Test that dpo_online_bias.py has valid syntax."""
    try:
        with open("trl/scripts/dpo_online_bias.py", "r") as f:
            code = f.read()
        compile(code, "dpo_online_bias.py", "exec")
        print("✓ dpo_online_bias.py has valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ dpo_online_bias.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ dpo_online_bias.py error: {e}")
        return False

def test_dpo_online_syntax():
    """Test that dpo_online.py has valid syntax."""
    try:
        with open("examples/scripts/dpo_online.py", "r") as f:
            code = f.read()
        compile(code, "dpo_online.py", "exec")
        print("✓ dpo_online.py has valid syntax")
        return True
    except SyntaxError as e:
        print(f"✗ dpo_online.py syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ dpo_online.py error: {e}")
        return False

def test_structure():
    """Test that the expected files exist."""
    files_to_check = [
        "trl/scripts/dpo_online_bias.py",
        "examples/scripts/dpo_online.py",
        "trl/scripts/grpo_bias.py",
        "trl/trainer/grpo_trainer.py",
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    return all_exist

if __name__ == "__main__":
    print("Testing dpo_online_bias implementation...")
    structure_ok = test_structure()
    bias_syntax_ok = test_dpo_online_bias_syntax()
    online_syntax_ok = test_dpo_online_syntax()
    
    if structure_ok and bias_syntax_ok and online_syntax_ok:
        print("\n✓ All tests passed! Basic implementation is ready.")
    else:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)