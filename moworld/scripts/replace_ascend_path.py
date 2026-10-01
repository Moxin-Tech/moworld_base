#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
"""
Batch-replace CANN/Ascend install paths in the MindSpeed-MM repository.

Background:
  In the international release, the CANN/HDK install path has changed from
  /usr/local/Ascend to /usr/local/npu. This script replaces all occurrences
  of the old path across the repository in one pass.

Usage:
  python3 scripts/replace_ascend_path.py [options]

Examples:
  # Preview changes without modifying any files
  python3 scripts/replace_ascend_path.py --dry-run

  # Apply replacement: /usr/local/Ascend -> /usr/local/npu (default)
  python3 scripts/replace_ascend_path.py
"""

import argparse
import os
import sys


# 需要处理的文件扩展名
DEFAULT_EXTENSIONS = {'.sh', '.md', '.rst', '.py'}
# 同时处理无扩展名的特殊文件名
SPECIAL_FILENAMES = {'Dockerfile'}


def is_target_file(filepath, extensions, special_filenames):
    """判断文件是否需要处理。"""
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    return ext in extensions or filename in special_filenames


def collect_files(scan_dir, extensions, special_filenames):
    """递归收集需要处理的文件列表。"""
    self_path = os.path.abspath(__file__)
    target_files = []
    for root, dirs, files in os.walk(scan_dir):
        # 跳过隐藏目录和常见无关目录
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
        for filename in files:
            filepath = os.path.join(root, filename)
            if os.path.abspath(filepath) == self_path:
                continue
            if is_target_file(filepath, extensions, special_filenames):
                target_files.append(filepath)
    return sorted(target_files)


def process_file(filepath, source_path, target_path, dry_run):
    """
    处理单个文件：将 source_path 替换为 target_path。
    返回 (changed, replacements_count)。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError):
        return False, 0

    if source_path not in content:
        return False, 0

    new_content = content.replace(source_path, target_path)
    count = content.count(source_path)

    if not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

    return True, count


def format_path(filepath, scan_dir):
    """返回相对于扫描目录的相对路径。"""
    try:
        return os.path.relpath(filepath, scan_dir)
    except ValueError:
        return filepath


def main():
    parser = argparse.ArgumentParser(
        description='Batch-replace CANN/Ascend install paths in the MindSpeed-MM repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--source',
        default='/usr/local/Ascend',
        help='Source path (default: /usr/local/Ascend)'
    )
    parser.add_argument(
        '--target',
        default='/usr/local/npu',
        help='Target path (default: /usr/local/npu)'
    )
    parser.add_argument(
        '--dir',
        default='.',
        help='Directory to scan (default: current directory)'
    )
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=None,
        help='File extension whitelist (default: .sh .md .rst .py). Example: --extensions .sh .md'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying any files'
    )

    args = parser.parse_args()

    scan_dir = os.path.abspath(args.dir)
    if not os.path.isdir(scan_dir):
        print(f"[ERROR] Directory not found: {scan_dir}", file=sys.stderr)
        sys.exit(1)

    if args.source == args.target:
        print("[ERROR] Source and target paths are identical, nothing to replace.", file=sys.stderr)
        sys.exit(1)

    extensions = set(args.extensions) if args.extensions else DEFAULT_EXTENSIONS

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Path replacement: {args.source} -> {args.target}")
    print(f"Scan directory : {scan_dir}")
    print(f"File types     : {', '.join(sorted(extensions))} + {', '.join(sorted(SPECIAL_FILENAMES))}")
    print("-" * 60)

    files = collect_files(scan_dir, extensions, SPECIAL_FILENAMES)
    print(f"Found {len(files)} candidate file(s), processing...\n")

    changed_files = []
    total_replacements = 0

    for filepath in files:
        changed, count = process_file(filepath, args.source, args.target, args.dry_run)
        if changed:
            rel_path = format_path(filepath, scan_dir)
            changed_files.append((rel_path, count))
            total_replacements += count
            action = "would replace" if args.dry_run else "replaced"
            print(f"  [{action} {count:3d}] {rel_path}")

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"[DRY RUN] {len(changed_files)} file(s) would be modified, {total_replacements} replacement(s) total.")
        print("          Remove --dry-run to apply changes.")
    else:
        print(f"[DONE] {len(changed_files)} file(s) modified, {total_replacements} replacement(s) total.")

    if not changed_files:
        print(f"No files containing '{args.source}' were found.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
