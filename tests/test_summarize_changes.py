# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import tempfile
import unittest
from pathlib import Path


class TestSummarizeChanges(unittest.TestCase):
    """Test the summarize_changes script functionality."""
    
    def test_script_exists(self):
        """Test that the summarize_changes script exists and is executable."""
        script_path = Path(__file__).parent.parent / "scripts" / "summarize_changes.py"
        self.assertTrue(script_path.exists(), "summarize_changes.py script should exist")
        
    def test_script_help(self):
        """Test that the script shows help correctly."""
        script_path = Path(__file__).parent.parent / "scripts" / "summarize_changes.py"
        result = subprocess.run(
            ["python", str(script_path), "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0, "Script should show help without error")
        self.assertIn("Summarize git changes", result.stdout)
        self.assertIn("--days", result.stdout)
        self.assertIn("--output", result.stdout)
        
    def test_script_runs_without_error(self):
        """Test that the script runs without error in a git repository."""
        script_path = Path(__file__).parent.parent / "scripts" / "summarize_changes.py"
        
        # Test with very short time period to avoid analyzing too much
        result = subprocess.run(
            ["python", str(script_path), "--days", "1"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        # Should not error, even if no commits found
        self.assertEqual(result.returncode, 0, f"Script should run without error. stderr: {result.stderr}")
        self.assertIn("Change Summary", result.stdout)
        
    def test_script_output_to_file(self):
        """Test that the script can output to a file."""
        script_path = Path(__file__).parent.parent / "scripts" / "summarize_changes.py"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            result = subprocess.run(
                ["python", str(script_path), "--days", "1", "--output", tmp_path],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            
            self.assertEqual(result.returncode, 0, f"Script should run without error. stderr: {result.stderr}")
            
            # Check that file was created and has content
            output_path = Path(tmp_path)
            self.assertTrue(output_path.exists(), "Output file should be created")
            
            content = output_path.read_text()
            self.assertIn("Change Summary", content)
            
        finally:
            # Clean up
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()


if __name__ == "__main__":
    unittest.main()