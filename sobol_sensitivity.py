import numpy as np
import pandas as pd

# ==========================================
# THE FIX: Force matplotlib to run in the background
import matplotlib
matplotlib.use('Agg') 
# ==========================================
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol

# Import your model and config
from agents.bacolod_model import BacolodModel
from agents.household_agent import HouseholdAgent
import barangay_config as config

def get_global_compliance(model):
    """Helper function to extract global compliance from the model."""
    households = [a for a in model.schedule.agents if isinstance(a, HouseholdAgent)]
    if not households: return 0.0
    compliant = sum(1 for h in households if h.is_compliant)
    return compliant / len(households)

def run_model_with_params(params, ticks=180):
    """
    Runs the simulation with a specific set of TPB parameters.
    We run for 180 ticks (2 Quarters) to allow social norms to settle.
    """
    w_a, w_sn, w_pbc, c_effort, decay = params

    # Create a custom TPB profile based on the SALib sample
    custom_profile = {
        "w_a": w_a,
        "w_sn": w_sn,
        "w_pbc": w_pbc,
        "c_effort": c_effort,
        "decay": decay
    }
    
    # Override all barangay profiles so the whole city uses these test parameters
    behavior_override = {key: custom_profile for key in config.BEHAVIOR_PROFILES.keys()}

    # Run the model headlessly (train_mode=True speeds it up)
    model = BacolodModel(train_mode=True, policy_mode="status_quo", behavior_override=behavior_override)
    
    for _ in range(ticks):
        if not model.running:
            break
        model.step()

    # Return the final compliance rate as the output metric
    return get_global_compliance(model)

if __name__ == "__main__":
    print("==================================================")
    print(" HOUSEHOLD AGENT SOBOL SENSITIVITY ANALYSIS")
    print("==================================================")

    # 1. DEFINE THE PROBLEM
    # These are the boundaries of human behavior we are testing.
    problem = {
        'num_vars': 5,
        'names': ['Weight_Attitude', 'Weight_SocialNorms', 'Weight_PBC', 'Cost_of_Effort', 'Decay_Rate'],
        'bounds': [
            [0.10, 0.50],   # Weight_Attitude
            [0.10, 0.50],   # Weight_SocialNorms
            [0.10, 0.50],   # Weight_PBC
            [0.20, 0.80],   # Cost_of_Effort (Physical/mental friction)
            [0.001, 0.010]  # Decay_Rate (How fast they forget)
        ]
    }

    # 2. GENERATE SAMPLES
    # N dictates how many samples to run. 
    # Total runs = N * (2 * num_vars + 2). 
    # N=32 results in 384 simulation runs. (Increase N to 128 or 256 for your final thesis export!)
    N = 32 
    param_values = saltelli.sample(problem, N)
    total_runs = len(param_values)
    
    print(f"Generated {total_runs} parameter combinations using Saltelli sampling.")
    print("Running simulations... This may take a few minutes depending on your CPU.")

    # 3. RUN SIMULATIONS
    Y = np.zeros([param_values.shape[0]])
    for i, X in enumerate(param_values):
        Y[i] = run_model_with_params(X, ticks=180) # Run for 2 quarters
        
        # Simple progress tracker
        if (i + 1) % 50 == 0 or (i + 1) == total_runs:
            print(f" > Completed {i + 1} / {total_runs} simulations...")

    print("\nSimulations complete! Analyzing Sobol Indices...")

    # 4. ANALYZE RESULTS
    Si = sobol.analyze(problem, Y, print_to_console=False)

    # Extract Data for Plotting
    names = problem['names']
    S1 = Si['S1']   # First-order sensitivity (Direct impact of the variable)
    ST = Si['ST']   # Total-order sensitivity (Impact including interactions with other variables)

    # 5. PRINT TO CONSOLE FOR THESIS TEXT
    print("\n--- SOBOL INDICES RESULTS ---")
    for i in range(len(names)):
        print(f"{names[i]}:")
        print(f"  First-Order (S1) : {S1[i]:.4f} (Direct influence)")
        print(f"  Total-Order (ST) : {ST[i]:.4f} (Influence including interactions)")

    # 6. GENERATE BAR CHART
    x = np.arange(len(names))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, S1, width, label='First-Order (S1)', color='#3498db')
    rects2 = ax.bar(x + width/2, ST, width, label='Total-Order (ST)', color='#e74c3c')

    ax.set_ylabel('Sensitivity Index')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the graph
    plt.savefig('sobol_sensitivity_results.svg', format='svg')
    print("\n[SUCCESS] Sensitivity graph saved as 'sobol_sensitivity_results.svg'")