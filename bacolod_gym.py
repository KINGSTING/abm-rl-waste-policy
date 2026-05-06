import gymnasium as gym
from gymnasium import spaces
import numpy as np
from agents.bacolod_model import BacolodModel
from agents.enforcement_agent import EnforcementAgent

class BacolodGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, policy_mode="HuDRL"):
        super(BacolodGymEnv, self).__init__()
        self.policy_mode = policy_mode
        # LEGAL: Maintaining the full 21 Municipal Levers
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(17,), dtype=np.float32)
        self.model = None
        self.prev_compliance = np.zeros(7, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # FIX: Only one initialization using the passed policy_mode
        self.model = BacolodModel(train_mode=True, policy_mode=self.policy_mode)
        obs = self.model.get_state()
        self.prev_compliance = obs[0:7].copy() 
        return obs, {}
    
    def _get_observation(self):
        if self.model is None: return np.zeros(17, dtype=np.float32)
        return self.model.get_state()

    def step(self, action):
        # 1. APPLY TEMPERATURE
        # This converts the raw PPO output into amplified 'desire'.
        # 10.0 makes the AI very decisive; 5.0 is more 'relaxed'.
        amplified = np.exp(action * 10.0)
        
        # 2. NORMALIZE TO RAW ACTION VECTOR
        # We calculate the 'Raw Intent' of the AI first.
        # We NO LONGER apply heuristics here; the Mayor will do that.
        total_desire = np.sum(amplified)
        action_vector = amplified / total_desire if total_desire > 0 else np.ones(21)/21.0
        
        # 3. EXECUTE INTERVENTION (THE HEURISTIC LAYER)
        # We pass the action_vector to the Mayor. 
        # If policy_mode is "HuDRL", the Mayor will modify this vector 
        # IN-PLACE using Graduation, Target Lock, and Phase-Shift.
        self.model.mayor.execute_intervention(action_vector)
        
        # 4. ADVANCE SIMULATION
        # Run for 90 ticks (one quarter).
        for _ in range(90):
            self.model.step() 
            if not self.model.running: 
                break

        # 5. GATHER OBSERVATIONS
        obs = self.model.get_state()
        curr_compliance = obs[0:7]
        
        # 6. CALCULATE REWARD
        # The reward is calculated based on the FINAL action_vector 
        # (after the Mayor's heuristics were applied).
        reward = self.calculate_reward(obs, action_vector, self.prev_compliance)
        
        # 7. UPDATE PREVIOUS COMPLIANCE & ADD BONUSES
        self.prev_compliance = curr_compliance.copy()
        avg_attitude = np.mean(obs[7:14])
        reward += (avg_attitude * 0.5)
        
        # 8. CHECK STOP CONDITIONS
        terminated = not self.model.running
        truncated = False
        if terminated and self.model.political_capital < 0.10: 
            reward -= 10.0
        
        info = {
            "quarter": self.model.quarter, 
            "compliance": np.mean(obs[0:7]), 
            "political_capital": obs[16]
        }
        
        return obs, float(reward), terminated, truncated, info

    def calculate_reward(self, obs, allocation_vector, prev_compliance):
        curr_compliance = obs[0:7] 
        political_capital = obs[16]
        budget_left = obs[14]
        
        # 1. Base reward for general compliance
        global_compliance = np.mean(curr_compliance)
        reward = global_compliance * 50.0  
        
        # 2. THE 70% SOCIAL NORM JACKPOT (Aligned with Graduation Rule)
        #if global_compliance >= 0.70: 
        #    reward += 200.0                   
            
        compliance_gains = curr_compliance - prev_compliance
        
        for i in range(7):
            start_idx = i * 3
            bgy_allocation = np.sum(allocation_vector[start_idx:start_idx+3])
            enf_idx = start_idx + 1
            
            # 3. THE "SHOCK AND SUSTAIN" BREADCRUMBS
            # Aligned from 0.90 to 0.70
            if curr_compliance[i] < 0.70:
                # Hint: Use enforcement to break the initial resistance
                reward += (allocation_vector[enf_idx] * 10.0) 
                
                # Jackpot: Reward the AI heavily when its actions actually increase compliance
                if compliance_gains[i] > 0:
                    jackpot = (bgy_allocation * compliance_gains[i]) * 100.0  
                    reward += jackpot
            else:
                # Hint: Once past 70%, switch to IEC and Incentives to lock it in
                sustain_budget = allocation_vector[start_idx] + allocation_vector[start_idx + 2]
                reward += (sustain_budget * 20.0) 
                
                # Efficiency: Prevent the AI from wasting too much money on a "solved" barangay
                if bgy_allocation > 0.10: 
                    reward -= (bgy_allocation * 10.0) 
        
        # 4. FOCUS ALLOCATION BONUS
        # Aligned from 0.90 to 0.70
        struggling_indices = np.where(curr_compliance < 0.70)[0]
        if len(struggling_indices) > 0:
            lowest_bgy_idx = struggling_indices[np.argmin(curr_compliance[struggling_indices])]
            focus_allocation = np.sum(allocation_vector[lowest_bgy_idx*3:(lowest_bgy_idx*3)+3])
            reward += (focus_allocation * 5.0)        
            
        # 5. RESOURCE & POLITICAL MANAGEMENT
        if budget_left <= 0.01: 
            reward -= 0.5                             
        
        # Political Capital is a gentle modifier, giving points for keeping approval high
        reward += (political_capital * 20.0)           

        return float(reward)