import sys
import os
import re
from pathlib import Path

def get_prefix_if_valid(folder_name):
    """
    Validates that the folder ends with __<number>.
    Returns the prefix (*) if valid, otherwise None.
    """
    # Matches everything before __ if followed by only digits at the end
    match = re.fullmatch(r"(.+)__(\d+)", folder_name)
    if match:
        return match.group(1)
    return None

def get_last_run_number(filepath):
    """Finds the highest '### Run n' number in the file."""
    if not filepath.exists():
        return -1
    content = filepath.read_text()
    runs = re.findall(r'### Run (\d+)', content)
    if not runs:
        return -1
    return max(map(int, runs))

def merge_raw_txt(src_file, dst_file):
    """Appends raw.txt with Run re-indexing."""
    if not src_file.exists():
        return
    
    if not dst_file.exists():
        print(f"{dst_file} does not exist")
        dst_file.write_text(src_file.read_text())
        return

    # Determine the starting number for the new runs
    offset = get_last_run_number(dst_file) + 1
    src_content = src_file.read_text()

    # Re-index '### Run n' to '### Run n+offset'
    def increment_run(match):
        run_num = int(match.group(1))
        return f"### Run {run_num + offset}"

    new_content = re.sub(r'### Run (\d+)', increment_run, src_content)

    with open(dst_file, 'a') as f:
        # Ensure a clean break between previous data and new data
        f.write('\n' + new_content.strip() + '\n')

def merge_statistics_csv(src_file, dst_file):
    """Appends statistics.csv content as a new block."""
    if not src_file.exists():
        return
    
    if not dst_file.exists():
        dst_file.write_text(src_file.read_text())
        return

    src_content = src_file.read_text().strip()
    with open(dst_file, 'a') as f:
        f.write(f"\n###\n{src_content}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_script.py <dst_exps_dir> <src_exps_dir>")
        sys.exit(1)

    dst_dir = Path(sys.argv[1])
    src_dir = Path(sys.argv[2])

    if not dst_dir.is_dir() or not src_dir.is_dir():
        print("Error: Both arguments must be valid directory paths.")
        sys.exit(1)

    # Map validated prefixes to the actual folder path in destination
    dst_map = {}
    for d in dst_dir.iterdir():
        if d.is_dir():
            prefix = get_prefix_if_valid(d.name)
            if prefix:
                dst_map[prefix] = d

    # Iterate through source experiments
    for src_exp in src_dir.iterdir():
        if not src_exp.is_dir():
            continue

        prefix = get_prefix_if_valid(src_exp.name)
        
        if not prefix:
            print(f"Skipping {src_exp.name}: Does not match pattern *__<number>")
            continue

        if prefix in dst_map:
            target_folder = dst_map[prefix]
            print(f"Merging Prefix [{prefix}]: {src_exp.name} -> {target_folder.name}")

            merge_raw_txt(src_exp / "raw.txt", target_folder / "raw.txt")
            merge_statistics_csv(src_exp / "statistics.csv", target_folder / "statistics.csv")
        else:
            print(f"No match in destination for prefix [{prefix}]. Skipping.")

if __name__ == "__main__":
    main()
