import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from stable_baselines3 import PPO
from bacolod_gym import BacolodGymEnv

# --- Configuration ---
MODEL_PATH = "models/ppo/bacolod_ppo_final.zip"
EPISODES = 5  # Independent seeds for statistical validation
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def calculate_cohens_d(group1, group2):
    """Calculates effect size (magnitude of the performance gap)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    # Handle edge case where variance is zero to avoid division by zero
    if var1 + var2 == 0: return 0.0
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def evaluate_policy(policy_type, agent=None):
    """Runs the environment across EPISODES seeds and records raw reward data."""
    env_mode = policy_type if policy_type in ["HuDRL", "Vanilla_DRL"] else "Vanilla_DRL"
    env = BacolodGymEnv(policy_mode=env_mode)
    raw_rewards = []

    print(f"Evaluating {policy_type}...")

    for seed in range(EPISODES):
        obs, _ = env.reset(seed=seed)
        done = False
        truncated = False
        ep_reward = 0.0
        
        while not (done or truncated):
            if agent and policy_type in ["HuDRL", "Vanilla_DRL"]:
                action, _ = agent.predict(obs, deterministic=True)
            elif policy_type == "Random":
                action = env.action_space.sample()
            elif policy_type == "Dirichlet":
                # Dirichlet ensures the budget sums to 1.0 automatically
                action = (np.random.dirichlet(np.ones(21)) * 2) - 1
            elif policy_type == "Greedy":
                # Logic: Find the weakest barangay and dump resources there
                compliance_rates = obs[:7] 
                weakest_idx = np.argmin(compliance_rates)
                action = np.full(21, -1.0)
                action[weakest_idx*3 : weakest_idx*3 + 3] = 1.0
            
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

        raw_rewards.append(ep_reward)

    return {
        "name": policy_type,
        "rewards_raw": raw_rewards,
        "mean": np.mean(raw_rewards),
        "std": np.std(raw_rewards)
    }

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    
    agent = PPO.load(MODEL_PATH)
    policy_names = ["Random", "Dirichlet", "Greedy", "Vanilla_DRL", "HuDRL"]
    results = [evaluate_policy(p, agent) for p in policy_names]

    # --- Statistical Calculations ---
    hu_res = next(r for r in results if r['name'] == "HuDRL")
    greedy_res = next(r for r in results if r['name'] == "Greedy")
    
    # Welch's T-Test: Best Baseline (Greedy) vs Proposed (HuDRL)
    t_stat, p_val = stats.ttest_ind(hu_res['rewards_raw'], greedy_res['rewards_raw'], equal_var=False)
    d_size = calculate_cohens_d(hu_res['rewards_raw'], greedy_res['rewards_raw'])

    # --- 1. Console Report ---
    print("\n" + "="*80)
    print(f"{'Algorithm':<15} | {'Mean Reward':<15} | {'Std Dev':<12} | {'Max Reward':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} | {r['mean']:<15.2f} | {r['std']:<12.2f} | {max(r['rewards_raw']):<12.2f}")
    print("="*80)
    print(f"HuDRL vs Greedy Statistical Rigor: p-value = {p_val:.5f}, Cohen's d = {d_size:.2f}")

    # --- 2. Save Data for Traceability ---
    export_df = pd.DataFrame({r['name']: r['rewards_raw'] for r in results})
    export_df.to_csv(os.path.join(RESULTS_DIR, "reward_raw_data.csv"), index=False)

    # --- 3. Reward-Focused Visualization (Box Plot) ---
    plt.style.use('seaborn-v0_8-muted') # Cleaner academic look
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [r['rewards_raw'] for r in results]
    labels = [r['name'] for r in results]
    
    # notch=True helps visualize confidence intervals for the median
    box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True)
    
    # IEEE-friendly color palette
    colors = ['#bdc3c7', '#bdc3c7', '#95a5a6', '#e74c3c', '#2ecc71']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add Significance Annotation
    if p_val < 0.05:
        x1, x2 = labels.index("Greedy") + 1, labels.index("HuDRL") + 1
        y_max = max([max(r['rewards_raw']) for r in results])
        y, h = y_max * 1.05, y_max * 0.02
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='black')
        plt.text((x1+x2)*.5, y+h, f"p={p_val:.4f} (Significant)", ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Total Episodic Reward ($R_t$)')
    plt.title(f'Policy Efficiency Comparison across {EPISODES} Seeds')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_rigor_boxplot.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()