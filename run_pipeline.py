import subprocess
import sys
import os
import time

def run_script(script_path):
    """Runs a single python script and checks for errors."""
    print(f"\n{'='*80}")
    print(f"Running: {script_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # Use the same python executable that is running this script
    cmd = [sys.executable, script_path]
    
    # Define log file path
    script_name = os.path.basename(script_path).replace('.py', '.log')
    log_path = os.path.join("outputs", "logs", script_name)
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    print(f"Logging to: {log_path}")

    # Run the script and wait for it to follow.
    # We pass a modified environment to force UTF-8 to handle unicode characters (like checkmarks) on Windows.
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    
    try:
        # Open log file for writing
        with open(log_path, 'w', encoding='utf-8') as log_file:
            # Run the script, redirecting stdout and stderr to the log file
            # We also tee to stdout if possible, or just log.
            # For simplicity, we'll redirect to log file and print a summary content later or allow user to tail it.
            # To keep it simple and robust: Redirect all to file.
            subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        
        duration = time.time() - start_time
        print(f"\n[SUCCESS] {script_path} completed in {duration:.2f} seconds.")
        print(f"Check {log_path} for details.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] execution failed for {script_path}.")
        print(f"Return code: {e.returncode}")
        print(f"Check {log_path} for error details.")
        sys.exit(1)
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] execution failed for {script_path}.")
        print(f"Return code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred while running {script_path}.")
        print(e)
        sys.exit(1)

def main():
    print("Starting Project Pipeline Execution...")
    print(f"Working Directory: {os.getcwd()}")
    
    # Define the sequence of scripts to run
    # Paths are relative to the project root
    scripts = [
        "workflow/01_data_setup.py",
        "workflow/02_baseline_reproduction.py",
        "workflow/03_methodology_test.py",
        "workflow/04_synthetic_experiment.py",
        "workflow/05_mnist_generators.py",
        "workflow/06_mnist_solver.py",
        "workflow/07_mnist_evaluation.py"
    ]
    
    total_start = time.time()
    
    for script in scripts:
        # Ensure path uses correct OS separators
        script_path = os.path.normpath(script)
        
        if not os.path.exists(script_path):
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)
            
        run_script(script_path)
        
    total_duration = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
