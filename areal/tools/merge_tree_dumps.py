#!/usr/bin/env python3
"""
Merge tree dump files from multiple ranks into a single file.

Usage:
    # Merge a specific call
    python -m areal.tools.merge_tree_dumps --dump-dir /path/to/dump --call 2
    
    # Merge all calls
    python -m areal.tools.merge_tree_dumps --dump-dir /data/tree/tree-data/tau2-16k-small --all
    
    # List available calls
    python -m areal.tools.merge_tree_dumps --dump-dir /data/tree/tree-data/tau2-16k-small --list

This will merge call2_rank0.pt, call2_rank1.pt, ... into call2.pt
"""

import argparse
import glob
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import torch

from areal.utils.data import concat_padded_tensors


def find_rank_files(dump_dir: str, call: int) -> List[Path]:
    """Find all rank files for a given call number."""
    pattern = os.path.join(dump_dir, f"call{call}_rank*.pt")
    files = sorted(glob.glob(pattern))
    return [Path(f) for f in files]


def find_trie_files(dump_dir: str, call: int) -> List[Path]:
    """Find all trie pickle files for a given call number."""
    pattern = os.path.join(dump_dir, f"call{call}_rank*_trie.pkl")
    files = sorted(glob.glob(pattern))
    return [Path(f) for f in files]


def merge_pt_files(pt_files: List[Path], output_path: Path, quiet: bool = False) -> None:
    """Merge multiple .pt dump files into a single file."""
    if not pt_files:
        raise ValueError("No .pt files to merge")
    
    if not quiet:
        print(f"Loading {len(pt_files)} rank files...")
    all_data = []
    for pt_file in pt_files:
        if not quiet:
            print(f"  Loading {pt_file.name}...")
        data = torch.load(pt_file, map_location="cpu")
        all_data.append(data)
    
    if not quiet:
        print("Merging tensor data with padding...")
    
    # Merge input_data using concat_padded_tensors (handles padding automatically)
    input_datas = [d["input_data"] for d in all_data]
    merged_input_data = concat_padded_tensors(input_datas, pad_value=0)
    
    # Concatenate all output_mbs lists
    merged_output_mbs = []
    for data in all_data:
        merged_output_mbs.extend(data["output_mbs"])
    
    merged_data = {
        "input_data": merged_input_data,
        "output_mbs": merged_output_mbs,
    }
    
    # Save merged data
    if not quiet:
        print(f"Saving merged tensor data to {output_path}...")
    torch.save(merged_data, output_path)
    
    # Print summary
    if not quiet:
        print(f"\nTensor Merge Summary:")
        print(f"  Input data shape: {merged_input_data['input_ids'].shape}")
        print(f"  Number of micro-batches: {len(merged_output_mbs)}")
        print(f"  Output file: {output_path}")


def merge_trie_files(trie_files: List[Path], output_path: Path, quiet: bool = False) -> None:
    """Merge multiple trie pickle files into a single file."""
    if not trie_files:
        raise ValueError("No trie files to merge")
    
    if not quiet:
        print(f"\nLoading {len(trie_files)} trie files...")
    all_tries = []
    for trie_file in trie_files:
        if not quiet:
            print(f"  Loading {trie_file.name}...")
        with open(trie_file, "rb") as f:
            tries = pickle.load(f)
            all_tries.extend(tries)
    
    # Save merged tries
    if not quiet:
        print(f"Saving merged trie data to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(all_tries, f)
    
    if not quiet:
        print(f"\nTrie Merge Summary:")
        print(f"  Total trees: {len(all_tries)}")
        print(f"  Output file: {output_path}")


def get_available_calls(dump_dir: Path) -> List[int]:
    """Get list of available call numbers in dump directory."""
    pattern = os.path.join(dump_dir, "call*_rank*.pt")
    files = glob.glob(pattern)
    
    # Extract unique call numbers
    calls = set()
    for f in files:
        basename = os.path.basename(f)
        # Extract call number from filename like "call2_rank0.pt"
        if basename.startswith("call") and "_rank" in basename:
            call_num = basename.split("_")[0].replace("call", "")
            if call_num.isdigit():
                calls.add(int(call_num))
    
    return sorted(calls)


def merge_single_call(dump_dir: Path, call: int, output_dir: Path, merge_trie: bool, quiet: bool = False) -> None:
    """Merge files for a single call number."""
    # Find rank files
    pt_files = find_rank_files(str(dump_dir), call)
    if not pt_files:
        if not quiet:
            print(f"Warning: No .pt files found for call={call}")
        return
    
    if not quiet:
        print(f"\n{'='*80}")
        print(f"Merging call={call} ({len(pt_files)} rank files)")
        print(f"{'='*80}")
        for f in pt_files:
            print(f"  {f.name}")
    
    # Merge .pt files
    output_pt = output_dir / f"call{call}.pt"
    merge_pt_files(pt_files, output_pt, quiet=quiet)
    
    # Merge trie files if requested
    if merge_trie:
        trie_files = find_trie_files(str(dump_dir), call)
        if trie_files:
            if not quiet:
                print(f"\nFound {len(trie_files)} trie files for call={call}")
            output_trie = output_dir / f"call{call}_trie.pkl"
            merge_trie_files(trie_files, output_trie, quiet=quiet)
        else:
            if not quiet:
                print(f"\nNo trie files found for call={call}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge tree dump files from multiple ranks into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available calls
  python -m areal.tools.merge_tree_dumps --dump-dir /path/to/dump --list
  
  # Merge call2 from all ranks
  python -m areal.tools.merge_tree_dumps --dump-dir /path/to/dump --call 2
  
  # Merge all calls
  python -m areal.tools.merge_tree_dumps --dump-dir /path/to/dump --all
  
  # Merge call5 and specify custom output directory
  python -m areal.tools.merge_tree_dumps --dump-dir /path/to/dump --call 5 --output-dir /path/to/output
  
  # Also merge trie files
  python -m areal.tools.merge_tree_dumps --dump-dir /path/to/dump --call 2 --merge-trie
        """,
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        required=True,
        help="Directory containing dump files",
    )
    parser.add_argument(
        "--call",
        type=int,
        help="Call number to merge (e.g., 2 for call2_rank*.pt)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Merge all available calls",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as dump-dir)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available call numbers",
    )
    parser.add_argument(
        "--merge-trie",
        action="store_true",
        default=False,
        help="Also merge trie pickle files (default: False, only merge tensor data)",
    )
    
    args = parser.parse_args()
    
    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        raise ValueError(f"Dump directory does not exist: {dump_dir}")
    
    # List mode
    if args.list:
        calls = get_available_calls(dump_dir)
        
        if not calls:
            print(f"No dump files found in {dump_dir}")
        else:
            print(f"Available call numbers in {dump_dir}:")
            for call in calls:
                pt_files = find_rank_files(str(dump_dir), call)
                trie_files = find_trie_files(str(dump_dir), call)
                print(f"  call={call}: {len(pt_files)} rank(s) (.pt), {len(trie_files)} rank(s) (.pkl)")
        return
    
    # Merge mode
    if args.call is None and not args.all:
        parser.error("Either --call or --all is required when not using --list")
    
    if args.call is not None and args.all:
        parser.error("Cannot specify both --call and --all")
    
    output_dir = Path(args.output_dir) if args.output_dir else dump_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge all calls
    if args.all:
        calls = get_available_calls(dump_dir)
        if not calls:
            print(f"No dump files found in {dump_dir}")
            return
        
        print(f"Found {len(calls)} call(s) to merge: {calls}")
        print(f"Output directory: {output_dir}")
        print(f"Merge trie files: {args.merge_trie}")
        
        for call in calls:
            merge_single_call(dump_dir, call, output_dir, args.merge_trie, quiet=False)
        
        print("\n" + "=" * 80)
        print(f"All merges completed! Processed {len(calls)} call(s)")
        print("=" * 80)
        return
    
    # Merge single call
    merge_single_call(dump_dir, args.call, output_dir, args.merge_trie, quiet=False)
    
    print("\n" + "=" * 80)
    print("Merge completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

