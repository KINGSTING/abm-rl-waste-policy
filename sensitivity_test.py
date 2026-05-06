import os
import sys
import warnings
import multiprocessing as mp
import pandas as pd
import numpy as np

# 1. THE KILL SWITCHES
os.environ['MPLBACKEND'] = 'Agg'
warnings.filterwarnings("ignore", message=".*Axes3D.*")

from agents.bacolod_model import BacolodModel

# Hardcoded Matrix: (T_Low, T_High)
TEST_MATRIX = [
    (0.20, 0.60), (0.25, 0.70), (0.30, 0.80), 
    (0.35, 0.60), (0.40, 0.70), (0.45, 0.80), 
    (0.50, 0.70), (0.55, 0.80), (0.60, 0.85)  
]

def run_hardcoded_sim(params):
    t_low, t_high = params
    pid = os.getpid()
    
    # FORCE output to terminal immediately
    sys.stdout.write(f"\n[CORE {pid}] STARTING: T_Low={t_low}, T_High={t_high}\n")
    sys.stdout.flush()
    
    model = BacolodModel(
        policy_mode="HuDRL", 
        train_mode=True, 
        sens_t_low=t_low, 
        sens_t_high=t_high
    )
    
    for step in range(1, 1081):
        model.step()
        
        # LOG PROGRESS EVERY 200 STEPS
        if step % 200 == 0:
            sys.stdout.write(f"[CORE {pid}] Progress: Step {step}/1080 (Scenario: {t_low}/{t_high})\n")
            sys.stdout.flush()
            
        if not model.running:
            break
        
    from agents.household_agent import HouseholdAgent
    agents = [a for a in model.schedule.agents if isinstance(a, HouseholdAgent)]
    compliance = sum(1 for a in agents if a.is_compliant) / len(agents) if agents else 0.0
    
    sys.stdout.write(f"[CORE {pid}] FINISHED: Compliance={compliance:.2%}\n")
    sys.stdout.flush()
    
    return {"t_low": t_low, "t_high": t_high, "final_compliance": compliance}

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")

    print("="*50)
    print("   HARDCODED ROBUSTNESS TEST: STEP-BY-STEP MONITORING")
    print("="*50)
    print(f"Distributing 9 scenarios across 4 cores...")

    results = []
    with mp.Pool(processes=4) as pool:
        # imap_unordered lets us process results as they come in
        for res in pool.imap_unordered(run_hardcoded_sim, TEST_MATRIX):
            results.append(res)
    
    # Save results
    df = pd.DataFrame(results).sort_values(by=['t_low', 't_high'])
    df.to_csv("results/robustness_test_results.csv", index=False)
    
    print("\n" + "="*50)
    print("FINAL SUMMARY TABLE")
    print("="*50)
    print(df.to_string(index=False))