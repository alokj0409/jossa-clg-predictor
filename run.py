import os
import sys
import subprocess
import time

def run_command(cmd, wait=False):
    print(f"Running command: {' '.join(cmd)}")
    if wait:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    else:
        return subprocess.Popen(cmd, shell=True)

def main():
    if len(sys.argv) < 2:
        print("JoSAA ML Predictor Orchestrator")
        print("Usage:")
        print("  python run.py pipeline      - Run data preprocessing & classification sampling")
        print("  python run.py train         - Train XGBoost models and generate choice lookups")
        print("  python run.py start         - Start both FastAPI backend and Streamlit frontend")
        print("  python run.py run-all       - Preprocess, train, and start servers sequentially")
        sys.exit(1)
        
    action = sys.argv[1]
    
    if action == "pipeline":
        run_command(["python", "src/data_pipeline.py"], wait=True)
        
    elif action == "train":
        run_command(["python", "src/train_models.py"], wait=True)
        
    elif action == "start":
        print("Starting FastAPI backend...")
        api_proc = run_command(["python", "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"])
        
        # Give API a moment to spin up
        time.sleep(3)
        
        print("Starting Streamlit frontend...")
        app_proc = run_command(["python", "-m", "streamlit", "run", "app/streamlit_app.py"])
        
        try:
            print("\nServers are running! Press Ctrl+C to stop them.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping servers...")
            api_proc.terminate()
            app_proc.terminate()
            
    elif action == "run-all":
        print("=== Step 1: Preprocessing Data ===")
        run_command(["python", "src/data_pipeline.py"], wait=True)
        
        print("\n=== Step 2: Training XGBoost Models ===")
        run_command(["python", "src/train_models.py"], wait=True)
        
        print("\n=== Step 3: Launching Servers ===")
        print("Starting FastAPI backend...")
        api_proc = run_command(["python", "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"])
        
        time.sleep(3)
        
        print("Starting Streamlit frontend...")
        app_proc = run_command(["python", "-m", "streamlit", "run", "app/streamlit_app.py"])
        
        try:
            print("\nServers are running! Press Ctrl+C to stop them.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping servers...")
            api_proc.terminate()
            app_proc.terminate()
            
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

if __name__ == "__main__":
    main()
