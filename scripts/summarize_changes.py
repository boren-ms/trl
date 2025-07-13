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

"""
Script to summarize git changes from the last month in detail.
Provides comprehensive analysis of commits, file changes, and statistics.
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


def make_parser(subparsers):
    """Create argument parser for the summarize-changes command."""
    parser = subparsers.add_parser(
        "summarize-changes", 
        help="Summarize git changes from the last month in detail"
    )
    parser.add_argument(
        "--days", 
        type=int, 
        default=30, 
        help="Number of days to look back (default: 30)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--repo-path", 
        type=str, 
        default=".", 
        help="Path to git repository (default: current directory)"
    )
    return parser


def run_git_command(command: List[str], cwd: str = None) -> str:
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=cwd
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running git command: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return ""


def get_commits_since_date(days_back: int = 30, repo_path: str = ".") -> List[Dict]:
    """Get all commits from the specified number of days back."""
    since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    # Get commit hashes and basic info
    commit_format = "--pretty=format:%H|%an|%ae|%ad|%s"
    command = [
        "git", "log", f"--since={since_date}", 
        commit_format, "--date=short"
    ]
    
    output = run_git_command(command, repo_path)
    if not output:
        return []
    
    commits = []
    for line in output.split('\n'):
        if line.strip():
            parts = line.split('|')
            if len(parts) >= 5:
                commits.append({
                    'hash': parts[0],
                    'author': parts[1],
                    'email': parts[2],
                    'date': parts[3],
                    'message': '|'.join(parts[4:])  # Rejoin in case message contains |
                })
    
    return commits


def get_commit_stats(commit_hash: str, repo_path: str = ".") -> Dict:
    """Get detailed statistics for a specific commit."""
    # Get file changes
    command = ["git", "show", "--name-status", commit_hash]
    output = run_git_command(command, repo_path)
    
    added_files = []
    modified_files = []
    deleted_files = []
    
    for line in output.split('\n'):
        if line.strip() and '\t' in line:
            status, filename = line.split('\t', 1)
            if status == 'A':
                added_files.append(filename)
            elif status == 'M':
                modified_files.append(filename)
            elif status == 'D':
                deleted_files.append(filename)
    
    # Get line changes
    command = ["git", "show", "--numstat", commit_hash]
    output = run_git_command(command, repo_path)
    
    total_additions = 0
    total_deletions = 0
    
    for line in output.split('\n'):
        if line.strip() and '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                total_additions += int(parts[0])
                total_deletions += int(parts[1])
    
    return {
        'added_files': added_files,
        'modified_files': modified_files,
        'deleted_files': deleted_files,
        'total_additions': total_additions,
        'total_deletions': total_deletions,
        'files_changed': len(added_files) + len(modified_files) + len(deleted_files)
    }


def categorize_changes(commits: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize commits based on their messages."""
    categories = {
        'features': [],
        'bugfixes': [],
        'documentation': [],
        'ci_cd': [],
        'refactoring': [],
        'tests': [],
        'dependencies': [],
        'other': []
    }
    
    for commit in commits:
        message = commit['message'].lower()
        
        # Feature patterns
        if any(keyword in message for keyword in ['feat', 'feature', 'add', 'new', 'implement']):
            categories['features'].append(commit)
        # Bug fix patterns
        elif any(keyword in message for keyword in ['fix', 'bug', 'patch', 'resolve', 'hotfix']):
            categories['bugfixes'].append(commit)
        # Documentation patterns
        elif any(keyword in message for keyword in ['doc', 'readme', 'comment', 'docs']):
            categories['documentation'].append(commit)
        # CI/CD patterns
        elif any(keyword in message for keyword in ['ci', 'build', 'deploy', 'workflow', 'action']):
            categories['ci_cd'].append(commit)
        # Refactoring patterns
        elif any(keyword in message for keyword in ['refactor', 'cleanup', 'style', 'format']):
            categories['refactoring'].append(commit)
        # Test patterns
        elif any(keyword in message for keyword in ['test', 'spec', 'coverage']):
            categories['tests'].append(commit)
        # Dependencies
        elif any(keyword in message for keyword in ['deps', 'dependency', 'requirements', 'bump']):
            categories['dependencies'].append(commit)
        else:
            categories['other'].append(commit)
    
    return categories


def get_file_type_stats(commits: List[Dict], repo_path: str = ".") -> Dict[str, Dict]:
    """Analyze changes by file type."""
    file_types = defaultdict(lambda: {'count': 0, 'additions': 0, 'deletions': 0})
    
    for commit in commits:
        # Get file-specific stats for this commit
        command = ["git", "show", "--numstat", commit['hash']]
        output = run_git_command(command, repo_path)
        
        for line in output.split('\n'):
            if line.strip() and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                    additions = int(parts[0])
                    deletions = int(parts[1])
                    filepath = parts[2]
                    
                    # Get file extension
                    path_obj = Path(filepath)
                    ext = path_obj.suffix.lower() if path_obj.suffix else 'no_extension'
                    
                    file_types[ext]['count'] += 1
                    file_types[ext]['additions'] += additions
                    file_types[ext]['deletions'] += deletions
    
    return dict(file_types)


def generate_summary_report(commits: List[Dict], repo_path: str = ".") -> str:
    """Generate a comprehensive summary report."""
    if not commits:
        return "# Change Summary - Last Month\n\nNo commits found in the last month.\n"
    
    # Get detailed stats for each commit
    all_stats = []
    for commit in commits:
        stats = get_commit_stats(commit['hash'], repo_path)
        commit['stats'] = stats
        all_stats.append(stats)
    
    # Categorize commits
    categories = categorize_changes(commits)
    
    # Calculate overall statistics
    total_additions = sum(stats['total_additions'] for stats in all_stats)
    total_deletions = sum(stats['total_deletions'] for stats in all_stats)
    total_files_changed = sum(stats['files_changed'] for stats in all_stats)
    
    # Get unique authors
    authors = list(set(commit['author'] for commit in commits))
    
    # Get file type statistics
    file_type_stats = get_file_type_stats(commits, repo_path)
    
    # Generate report
    report = []
    report.append("# Change Summary - Last Month")
    report.append("")
    report.append(f"**Analysis Period:** {(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
    report.append("")
    
    # Overall statistics
    report.append("## 📊 Overall Statistics")
    report.append("")
    report.append(f"- **Total Commits:** {len(commits)}")
    report.append(f"- **Contributors:** {len(authors)}")
    report.append(f"- **Files Changed:** {total_files_changed}")
    report.append(f"- **Lines Added:** {total_additions:,}")
    report.append(f"- **Lines Deleted:** {total_deletions:,}")
    report.append(f"- **Net Change:** {total_additions - total_deletions:+,} lines")
    report.append("")
    
    # Contributors
    if authors:
        report.append("## 👥 Contributors")
        report.append("")
        author_stats = defaultdict(lambda: {'commits': 0, 'additions': 0, 'deletions': 0})
        for commit in commits:
            author_stats[commit['author']]['commits'] += 1
            author_stats[commit['author']]['additions'] += commit['stats']['total_additions']
            author_stats[commit['author']]['deletions'] += commit['stats']['total_deletions']
        
        for author in sorted(authors):
            stats = author_stats[author]
            report.append(f"- **{author}**: {stats['commits']} commits, +{stats['additions']:,}/-{stats['deletions']:,} lines")
        report.append("")
    
    # Changes by category
    report.append("## 📁 Changes by Category")
    report.append("")
    
    category_names = {
        'features': '🚀 New Features',
        'bugfixes': '🐛 Bug Fixes',
        'documentation': '📚 Documentation',
        'ci_cd': '🔧 CI/CD',
        'refactoring': '♻️ Refactoring',
        'tests': '🧪 Tests',
        'dependencies': '📦 Dependencies',
        'other': '🔖 Other'
    }
    
    for category, commits_in_category in categories.items():
        if commits_in_category:
            report.append(f"### {category_names[category]} ({len(commits_in_category)} commits)")
            report.append("")
            for commit in commits_in_category:
                stats = commit['stats']
                report.append(f"- **{commit['hash'][:8]}** by {commit['author']} ({commit['date']})")
                report.append(f"  - {commit['message']}")
                report.append(f"  - Files changed: {stats['files_changed']}, +{stats['total_additions']}/-{stats['total_deletions']} lines")
                report.append("")
    
    # File type analysis
    if file_type_stats:
        report.append("## 📄 Changes by File Type")
        report.append("")
        sorted_types = sorted(file_type_stats.items(), key=lambda x: x[1]['count'], reverse=True)
        for ext, stats in sorted_types[:10]:  # Show top 10
            ext_name = ext if ext != 'no_extension' else 'No extension'
            report.append(f"- **{ext_name}**: {stats['count']} files, +{stats['additions']:,}/-{stats['deletions']:,} lines")
        report.append("")
    
    # Recent commits
    report.append("## 📝 Recent Commits (Last 10)")
    report.append("")
    recent_commits = sorted(commits, key=lambda x: x['date'], reverse=True)[:10]
    for commit in recent_commits:
        stats = commit['stats']
        report.append(f"- **{commit['hash'][:8]}** `{commit['date']}` by {commit['author']}")
        report.append(f"  {commit['message']}")
        if stats['files_changed'] > 0:
            report.append(f"  *{stats['files_changed']} files, +{stats['total_additions']}/-{stats['total_deletions']} lines*")
        report.append("")
    
    return '\n'.join(report)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(
            description="Summarize git changes from the last month in detail"
        )
        parser.add_argument(
            "--days", 
            type=int, 
            default=30, 
            help="Number of days to look back (default: 30)"
        )
        parser.add_argument(
            "--output", 
            type=str, 
            help="Output file path (default: print to stdout)"
        )
        parser.add_argument(
            "--repo-path", 
            type=str, 
            default=".", 
            help="Path to git repository (default: current directory)"
        )
        
        args = parser.parse_args()
    
    # Verify we're in a git repository
    try:
        run_git_command(["git", "rev-parse", "--git-dir"], args.repo_path)
    except:
        print("Error: Not in a git repository or git not available")
        sys.exit(1)
    
    print(f"Analyzing changes from the last {args.days} days...")
    
    # Get commits
    commits = get_commits_since_date(args.days, args.repo_path)
    
    # Generate report
    report = generate_summary_report(commits, args.repo_path)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()