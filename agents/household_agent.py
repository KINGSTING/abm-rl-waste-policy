from xml.parsers.expat import model

import mesa
import random
import numpy as np

class HouseholdAgent(mesa.Agent):
    """
    Household Agent based on Theory of Planned Behavior (TPB).
    Implements a 'Two-Stage Social Shield' and Multi-Layered Governance Interactions.
    """
    def __init__(self, unique_id, model, income_level, initial_compliance, behavior_params=None):
        super().__init__(unique_id, model)
        self.income_level = income_level
        self.is_compliant = initial_compliance
        self.barangay = None    
        self.barangay_id = None 

        if behavior_params is None:
            behavior_params = {"w_a": 0.4, "w_sn": 0.3, "w_pbc": 0.3, "c_effort": 0.2, "decay": 0.005}

        self.w_a = behavior_params["w_a"]
        self.w_sn = behavior_params["w_sn"]
        self.w_pbc = behavior_params["w_pbc"]
        self.c_effort_base = behavior_params["c_effort"]
        self.attitude_decay_rate = behavior_params["decay"]

        # Seeding based on initial state
        if self.is_compliant:
            self.attitude = random.uniform(0.75, 0.90) 
            self.sn = random.uniform(0.6, 0.9)
            self.pbc = random.uniform(0.6, 0.9)
        else:
            # REVERTED: Give them enough TPB buffer so they don't instantly decay to 0.
            self.attitude = random.uniform(0.1, 0.4)
            self.sn = random.uniform(0.1, 0.4)
            self.pbc = random.uniform(0.1, 0.4)

        self.utility = 0.0
        self.redeemed_this_quarter = False
        self.perceived_unfairness = False
        self.days_since_fined = 999 # <--- ADD THIS
        
        # Pull dynamic effort base from the model if it exists (for Sobol)
        if hasattr(model, "c_effort_base_override"):
            self.c_effort_base = model.c_effort_base_override
        else:
            self.c_effort_base = behavior_params["c_effort"]
        
        # Pull dynamic thresholds
        self.t_high = getattr(model, "threshold_high", 0.70)
        self.t_low = getattr(model, "threshold_low", 0.40)

    def update_social_norms(self):
        # 1. Calculate Local Compliance
        # --- FAST MATH OVERRIDE ---
        if self.model.train_mode:
            local_compliance = self.barangay.compliance_rate if self.barangay else 0.0
        else:
            # Slow visual grid logic (Only used for Server/UI)
            neighbors = self.model.grid.get_neighbors(self.pos, moore=True, radius=2)
            household_neighbors = [n for n in neighbors if isinstance(n, HouseholdAgent) and n.barangay_id == self.barangay_id]
            
            local_compliance = 0.0
            if household_neighbors:
                compliant_count = sum(1 for n in household_neighbors if n.is_compliant)
                local_compliance = compliant_count / len(household_neighbors)

        # Authority Boost
        authority_boost = 0.0
        if self.barangay:
            authority_boost = self.barangay.enforcement_intensity * 0.50

        # Perception Cap 
        target_sn = min(0.92, local_compliance + authority_boost)

        # --- 2. Apply Social Shield (Inertia + Tipping Point) ---
        current_bgy_compliance = self.barangay.compliance_rate if self.barangay else 0.0

        # --- Apply Social Shield ---
        # Look directly at the model to ensure the sensitivity values are used
        t_high = getattr(self.model, "threshold_high", 0.70)
        t_low = getattr(self.model, "threshold_low", 0.40)

        if target_sn < self.sn: 
            if current_bgy_compliance > t_high:
                self.sn = (0.95 * self.sn) + (0.05 * target_sn)
            elif current_bgy_compliance > t_low:
                self.sn = (0.85 * self.sn) + (0.15 * target_sn) 
            else:
                self.sn = target_sn
        else:
            # GROWTH MODE (The Tipping Point) - WAS MISSING
            if current_bgy_compliance > t_low:
                self.sn = min(1.0, target_sn * 1.30) # Amplified social pressure
            else:
                self.sn = target_sn

    def update_attitude(self):
        current_compliance = self.barangay.compliance_rate if self.barangay else 0.0
        
        # 1. Decay with Shielding
        decay_damper = 1.0
        if current_compliance > self.t_high:
            decay_damper = 0.05 
        elif current_compliance > self.t_low:
            decay_damper = 0.50 
        
        self.attitude -= (self.attitude_decay_rate * decay_damper)

        # 2. IEC Synergy
        if self.barangay:
            iec = getattr(self.barangay, 'iec_intensity', 0)
            enf = getattr(self.barangay, 'enforcement_intensity', 0)
            # BUFFED: Divided by 30 instead of 90. Spreads the boost over a month instead of a day.
            boost = (iec * 0.025 * (1.0 + (enf * 3.0))) / 30.0
            self.attitude += boost

        # 3. Pride & Complacency
        if current_compliance > 0.70:
            if self.attitude < 0.80:
                self.attitude += 0.003  # Community Pride
            if self.attitude > 0.90:
                self.attitude -= random.uniform(0.005, 0.015) # Complacency Tax

        self.attitude = max(0.0, min(0.95, self.attitude))
        
    def make_decision(self):
        """
        Calculates Utility based on Monetary factors and TPB.
        """
        if not self.barangay: return

        # 1. Variable Decay based on Reward History
        day_of_quarter = (self.model.tick % 90) + 1
        current_decay = self.attitude_decay_rate
        
        if day_of_quarter < 20:
            if self.perceived_unfairness:
                current_decay *= 3.0  # Resentment
            elif self.redeemed_this_quarter:
                current_decay *= 2.0  # Satiation
        self.attitude -= current_decay

        # 2. Economic Impact (Temporal Discounting)
        gamma = 1.5 if self.income_level == 1 else (1.0 if self.income_level == 2 else 0.8)
        fine = self.barangay.fine_amount
        prob_detection = self.barangay.enforcement_intensity
        
        # Expected Daily Incentive
        prob_of_getting_picked = 0.02 
        daily_incentive = (self.barangay.incentive_val * prob_of_getting_picked) / 90.0
        
        monetary_impact = daily_incentive + (fine * prob_detection)
        
        # =================================================================
        # THE HABIT SHIELD FIX (The Staircase Lock-in)
        # When compliance crosses the threshold, the physical and mental 
        # "cost of effort" drops permanently because it becomes a habit.
        # =================================================================
        current_compliance = self.barangay.compliance_rate if self.barangay else 0.0
        effort_discount = 0.0
        
        if current_compliance >= self.t_high: # Use dynamic variable
            effort_discount = self.c_effort_base * 0.85  
        elif current_compliance >= self.t_low: # Use dynamic variable
            effort_discount = self.c_effort_base * 0.40

        # Apply the discount to the base effort
        c_net = (self.c_effort_base - effort_discount) - (gamma * monetary_impact / 2000.0) 
        # =================================================================

        # 3. TPB Utility Summation
        epsilon = random.gauss(0, 0.05)
        self.utility = (self.w_a * self.attitude) + \
                       (self.w_sn * self.sn) + \
                       (self.w_pbc * self.pbc) - \
                       c_net + epsilon

        # --- THE FEAR MEMORY FIX ---
        self.days_since_fined += 1
        if self.days_since_fined < 30:
            self.is_compliant = True  # Too scared to break the law for a month
        else:
            self.is_compliant = (self.utility > 0.0)
            # Human Error (1% Rule)
            if self.is_compliant and random.random() < 0.01:
                 self.is_compliant = False

    def attempt_redemption(self):
        # Reset at the very start of the quarter
        if self.model.tick % 90 == 0:
            self.redeemed_this_quarter = False
            self.perceived_unfairness = False

        if self.is_compliant and not self.redeemed_this_quarter and self.barangay:
            # 10% daily chance to visit the office
            if random.random() < 0.10: 
                reward_amount = self.barangay.incentive_val
                if reward_amount > 0:
                    # give_reward now accesses both Barangay Base and LGU Universal Pools
                    success = self.barangay.give_reward(reward_amount)
                    self.redeemed_this_quarter = True 

                    if success:
                        self.attitude = min(0.95, self.attitude + 0.05)
                    else:
                        # Bankruptcy/Unfairness Penalty
                        self.attitude -= 0.30  
                        self.is_compliant = False
                        self.perceived_unfairness = True

    def receive_incentive(self, amount):
        """
        Logic for when the Barangay pushes an incentive to the household.
        """
        self.attitude = min(0.95, self.attitude + 0.05) 
        self.redeemed_this_quarter = True

    def get_fined(self, amount):
        """
        Triggered dynamically by an EnforcementAgent passing its specific fine_amount.
        (e.g., 500 for local Tanod, 1000 for Municipal Inspector).
        """
        # --- 1. TRACK THE FINE ---
        self.is_fined = True
        if not hasattr(self, 'fine_amount'):
            self.fine_amount = 0
        self.fine_amount += amount

        # =================================================================
        # 2. THE POLITICAL BACKLASH MECHANISM (FIXED)
        # The Mayor ONLY gets blamed for Municipal-level crackdowns (fines > 500).
        # They do not lose capital for regular daily Barangay Tanod fines.
        # =================================================================
        if amount > 500:
            political_penalty = 0.00010  # Heavier penalty, but only applies to Mayor's actions
            self.model.political_capital -= political_penalty
            self.model.political_capital = max(0.0, self.model.political_capital)

        # --- 3. Direct Economic Hit (Scaled by Income level) ---
        gamma = 1.5 if getattr(self, 'income_level', 2) == 1 else (1.0 if getattr(self, 'income_level', 2) == 2 else 0.8)
        econ_penalty = (amount / 1000.0) * gamma
        self.utility -= econ_penalty
        
        # --- 4. Resentment (Drop in attitude) ---
        self.attitude = max(0.0, self.attitude - 0.10)
        
        # --- 5. Immediate Behavioral Correction (Fear Factor) ---
        import random
        fear_probability = 0.40 if amount <= 500 else 0.60
        if random.random() < fear_probability:
            self.is_compliant = True
            self.pbc = min(0.95, self.pbc + 0.05)
            self.days_since_fined = 0

    def step(self):
        self.update_attitude()
        self.update_social_norms()
        self.make_decision()
        self.attempt_redemption()