#!/usr/bin/env python3
"""
Simple test script for DPO online phi4-multimodal training.
This script tests the basic functionality without requiring actual model weights or datasets.
"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch
import torch

# Add the trl package to path
sys.path.insert(0, '/home/runner/work/trl/trl')

def test_script_imports():
    """Test that the script can be imported without errors."""
    try:
        import examples.scripts.dpo_online_phi4_multimodal as script
        print("✓ Script imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import script: {e}")
        return False

def test_multimodal_script_arguments():
    """Test the custom ScriptArguments class."""
    try:
        from examples.scripts.dpo_online_phi4_multimodal import MultimodalScriptArguments
        args = MultimodalScriptArguments()
        assert hasattr(args, 'dataset_config_path')
        assert hasattr(args, 'tsv_paths')
        assert hasattr(args, 'max_train_samples')
        print("✓ MultimodalScriptArguments class works correctly")
        return True
    except Exception as e:
        print(f"✗ Failed to test MultimodalScriptArguments: {e}")
        return False

def test_audio_dataset_creation():
    """Test the audio dataset creation function."""
    try:
        from examples.scripts.dpo_online_phi4_multimodal import create_multimodal_dataset
        from unittest.mock import Mock
        
        # Mock the script arguments
        mock_script_args = Mock()
        mock_script_args.dataset_name = "openasr"
        mock_script_args.max_train_samples = 10
        mock_script_args.dataset_streaming = False
        mock_script_args.dataset_config_path = None
        
        mock_training_args = Mock()
        
        # Mock the create_audio_dataset function to avoid actually loading data
        with patch('examples.scripts.dpo_online_phi4_multimodal.create_audio_dataset') as mock_create:
            mock_dataset = Mock()
            mock_create.return_value = mock_dataset
            
            result = create_multimodal_dataset(mock_script_args, mock_training_args)
            
            # Verify the function was called with correct parameters
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            assert call_args['dataset_name'] == 'openasr'
            assert call_args['load_audio'] == True
            
        print("✓ Dataset creation function works correctly")
        return True
    except Exception as e:
        print(f"✗ Failed to test dataset creation: {e}")
        return False

def test_command_line_interface():
    """Test that the command line interface works."""
    try:
        import subprocess
        import sys
        
        # Test help command
        result = subprocess.run([
            sys.executable, 'examples/scripts/dpo_online_phi4_multimodal.py', '--help'
        ], capture_output=True, text=True, cwd='/home/runner/work/trl/trl')
        
        if result.returncode == 0 and 'dataset_config_path' in result.stdout:
            print("✓ Command line interface works correctly")
            return True
        else:
            print(f"✗ Command line interface failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Failed to test command line interface: {e}")
        return False

def test_config_file():
    """Test the configuration file."""
    try:
        import json
        config_path = '/home/runner/work/trl/trl/examples/configs/audio_dataset_config.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Verify required fields are present
        required_fields = ['dataset_name', 'load_audio', 'biasing', 'simu_perference']
        for field in required_fields:
            assert field in config, f"Missing required field: {field}"
        
        print("✓ Configuration file is valid")
        return True
    except Exception as e:
        print(f"✗ Failed to test configuration file: {e}")
        return False

def main():
    """Run all tests."""
    print("Running DPO Online Phi4-MultiModal Tests...")
    print("=" * 50)
    
    tests = [
        test_script_imports,
        test_multimodal_script_arguments,
        test_audio_dataset_creation,
        test_command_line_interface,
        test_config_file,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1

if __name__ == '__main__':
    sys.exit(main())