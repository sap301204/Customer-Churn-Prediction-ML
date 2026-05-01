import os
import subprocess
import sys

def run(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    if not os.path.exists("data/churn_frame.csv"):
        run(f"{sys.executable} src/generate_data.py --rows 5000")
    run(f"{sys.executable} src/train_model.py")

    if os.path.exists("data/telco_customer_churn.csv"):
        run(f"{sys.executable} src/train_telco_model.py")
    else:
        print("IBM Telco file not found. Add data/telco_customer_churn.csv to train Telco model.")
