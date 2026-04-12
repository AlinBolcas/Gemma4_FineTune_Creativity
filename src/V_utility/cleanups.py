#!/usr/bin/env python3
"""
Cleanup utility to remove Python cache files and directories.
Removes __pycache__ directories and .pyc files from the project.
"""

import os
import shutil
from pathlib import Path
from typing import List

# Try to import Utils for colored output
try:
    from src.VI_utils.utils import Utils
    USE_COLORED_OUTPUT = True
except ImportError:
    try:
        # Fallback: try importing Utils dynamically
        import sys
        import importlib.util
        
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        utils_path = project_root / "src" / "VI_utils" / "utils.py"
        
        if utils_path.exists():
            spec = importlib.util.spec_from_file_location("utils", str(utils_path))
            if spec and spec.loader:
                utils_module = importlib.util.module_from_spec(spec)
                sys.modules["utils"] = utils_module
                spec.loader.exec_module(utils_module)
                Utils = utils_module.Utils
                USE_COLORED_OUTPUT = True
        else:
            USE_COLORED_OUTPUT = False
    except Exception:
        USE_COLORED_OUTPUT = False


def get_project_root() -> Path:
    """Find the project root by looking for main.py or .git"""
    current_dir = Path(__file__).resolve().parent.parent.parent
    
    while current_dir != current_dir.parent:
        if (current_dir / "main.py").exists() or (current_dir / ".git").is_dir():
            return current_dir
        current_dir = current_dir.parent
    
    # Fallback: assume we're in src/VI_utils/cleanups.py -> project root is 3 levels up
    return Path(__file__).resolve().parent.parent.parent


def find_pycache_dirs(root_dir: Path) -> List[Path]:
    """Find all __pycache__ directories recursively."""
    pycache_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip common ignore dirs
        dirs[:] = [d for d in dirs if d not in {'.git', 'venv', '.venv', '.venv_mac', 'node_modules', '.idea', '.vscode'}]
        
        if '__pycache__' in dirs:
            pycache_path = Path(root) / '__pycache__'
            pycache_dirs.append(pycache_path)
    
    return pycache_dirs


def find_pyc_files(root_dir: Path) -> List[Path]:
    """Find all .pyc files recursively."""
    pyc_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip common ignore dirs
        dirs[:] = [d for d in dirs if d not in {'.git', 'venv', '.venv', '.venv_mac', 'node_modules', '.idea', '.vscode'}]
        
        for file in files:
            if file.endswith('.pyc'):
                pyc_files.append(Path(root) / file)
    
    return pyc_files


def remove_pycache_dirs(pycache_dirs: List[Path], dry_run: bool = False) -> int:
    """Remove __pycache__ directories."""
    removed_count = 0
    
    for pycache_dir in pycache_dirs:
        if dry_run:
            if USE_COLORED_OUTPUT:
                Utils.printColoured(f"[DRY RUN] Would remove: {pycache_dir}", "yellow")
            else:
                print(f"[DRY RUN] Would remove: {pycache_dir}")
        else:
            try:
                shutil.rmtree(pycache_dir)
                if USE_COLORED_OUTPUT:
                    Utils.printColoured(f"✅ Removed: {pycache_dir}", "green")
                else:
                    print(f"✅ Removed: {pycache_dir}")
                removed_count += 1
            except Exception as e:
                if USE_COLORED_OUTPUT:
                    Utils.printColoured(f"❌ Failed to remove {pycache_dir}: {e}", "red")
                else:
                    print(f"❌ Failed to remove {pycache_dir}: {e}")
    
    return removed_count


def remove_pyc_files(pyc_files: List[Path], dry_run: bool = False) -> int:
    """Remove .pyc files."""
    removed_count = 0
    
    for pyc_file in pyc_files:
        if dry_run:
            if USE_COLORED_OUTPUT:
                Utils.printColoured(f"[DRY RUN] Would remove: {pyc_file}", "yellow")
            else:
                print(f"[DRY RUN] Would remove: {pyc_file}")
        else:
            try:
                pyc_file.unlink()
                if USE_COLORED_OUTPUT:
                    Utils.printColoured(f"✅ Removed: {pyc_file}", "green")
                else:
                    print(f"✅ Removed: {pyc_file}")
                removed_count += 1
            except Exception as e:
                if USE_COLORED_OUTPUT:
                    Utils.printColoured(f"❌ Failed to remove {pyc_file}: {e}", "red")
                else:
                    print(f"❌ Failed to remove {pyc_file}: {e}")
    
    return removed_count


def cleanup_pycache(dry_run: bool = False, remove_pyc: bool = True) -> dict:
    """
    Clean up all __pycache__ directories and optionally .pyc files.
    
    Args:
        dry_run: If True, only show what would be removed without actually removing
        remove_pyc: If True, also remove .pyc files (default: True)
    
    Returns:
        Dictionary with cleanup statistics
    """
    project_root = get_project_root()
    
    if USE_COLORED_OUTPUT:
        Utils.printColoured(f"\n🔍 Scanning project root: {project_root}", "cyan")
    else:
        print(f"\n🔍 Scanning project root: {project_root}")
    
    # Find all __pycache__ directories
    pycache_dirs = find_pycache_dirs(project_root)
    
    # Find all .pyc files
    pyc_files = []
    if remove_pyc:
        pyc_files = find_pyc_files(project_root)
    
    # Print summary
    if USE_COLORED_OUTPUT:
        Utils.printColoured(f"\n📊 Found {len(pycache_dirs)} __pycache__ directories", "cyan")
        if remove_pyc:
            Utils.printColoured(f"📊 Found {len(pyc_files)} .pyc files", "cyan")
    else:
        print(f"\n📊 Found {len(pycache_dirs)} __pycache__ directories")
        if remove_pyc:
            print(f"📊 Found {len(pyc_files)} .pyc files")
    
    if dry_run:
        if USE_COLORED_OUTPUT:
            Utils.printColoured("\n🔍 DRY RUN MODE - No files will be removed", "yellow")
        else:
            print("\n🔍 DRY RUN MODE - No files will be removed")
    
    # Remove __pycache__ directories
    removed_dirs = remove_pycache_dirs(pycache_dirs, dry_run)
    
    # Remove .pyc files
    removed_pyc = 0
    if remove_pyc:
        removed_pyc = remove_pyc_files(pyc_files, dry_run)
    
    # Summary
    stats = {
        'pycache_dirs_found': len(pycache_dirs),
        'pycache_dirs_removed': removed_dirs,
        'pyc_files_found': len(pyc_files),
        'pyc_files_removed': removed_pyc,
        'dry_run': dry_run
    }
    
    if USE_COLORED_OUTPUT:
        Utils.printColoured("\n" + "=" * 60, "cyan")
        Utils.printColoured("📊 CLEANUP SUMMARY", "cyan")
        Utils.printColoured("=" * 60, "cyan")
        Utils.printColoured(f"✅ Removed {removed_dirs} __pycache__ directories", "green")
        if remove_pyc:
            Utils.printColoured(f"✅ Removed {removed_pyc} .pyc files", "green")
        Utils.printColoured("=" * 60 + "\n", "cyan")
    else:
        print("\n" + "=" * 60)
        print("📊 CLEANUP SUMMARY")
        print("=" * 60)
        print(f"✅ Removed {removed_dirs} __pycache__ directories")
        if remove_pyc:
            print(f"✅ Removed {removed_pyc} .pyc files")
        print("=" * 60 + "\n")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up Python cache files and directories")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without actually removing files'
    )
    parser.add_argument(
        '--no-pyc',
        action='store_true',
        help='Only remove __pycache__ directories, keep .pyc files'
    )
    
    args = parser.parse_args()
    
    cleanup_pycache(dry_run=args.dry_run, remove_pyc=not args.no_pyc)

