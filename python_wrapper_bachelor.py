import subprocess
import re
import os
import shutil
import glob

# Path to executable and its working directory
EXEC = "/home/shahin1/CLionProjects/bachelor_thesis/cmake-build-debug/bachelor_thesis"
CWD  = "/home/shahin1/CLionProjects/bachelor_thesis/cmake-build-debug"

RUNS = 20

best_score = 0
best_run = None
worst_score = 100
worst_run = None
scores = []

# Directory to save all run outputs
os.makedirs("runs", exist_ok=True)

# List of known CSV patterns to look for
csv_patterns = [
    "/home/shahin1/CLionProjects/bachelor_thesis/detected_subspaces_experiment2.csv",
    "/home/shahin1/CLionProjects/bachelor_thesis/qdf_representation_experiment2.csv",
    "/home/shahin1/CLionProjects/bachelor_thesis/mean_qdf_experiment2.csv",
    "/home/shahin1/CLionProjects/bachelor_thesis/clustered_data_experiment2.csv"
]

for i in range(1, RUNS + 1):
    print(f"=== RUN {i} ===")

    # Run the C++ executable in batch mode
    log = subprocess.check_output([EXEC, "--novis"], text=True, cwd=CWD)

    # Extract metric
    m = re.search(r"Global accuracy:\s*([0-9.]+)", log)
    if not m:
        print("⚠️ No metric found")
        continue

    score = float(m.group(1))
    print("score:", score)
    scores.append(score)

    # Make folder for this run
    run_dir = f"runs/run_{i}"
    os.makedirs(run_dir, exist_ok=True)

    # Save log
    with open(os.path.join(run_dir, "log.txt"), "w") as f:
        f.write(log)

    # Copy all CSVs matching patterns
    for pattern in csv_patterns:
        # Resolve relative path pattern
        full_pattern = os.path.join(CWD, pattern)
        for src_path in glob.glob(full_pattern):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(run_dir, fname)
            shutil.copy(src_path, dst_path)

    # Track best/worst scores
    if score > best_score:
        best_score = score
        best_run = i

    if score < worst_score:
        worst_score = score
        worst_run = i

print("=== RESULTS ===")
print("Best run:", best_run, "score =", best_score)
print("Worst run:", worst_run, "score =", worst_score)
