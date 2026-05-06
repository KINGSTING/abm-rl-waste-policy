import mesa
import numpy as np
import random
from agents.enforcement_agent import EnforcementAgent
from agents.household_agent import HouseholdAgent

class MayorAgent(mesa.Agent):
    """
    The Mayor Agent is the DRL-driven Executive.
    It manages the Municipal SWM Fund to perform 'Direct Intervention' 
    on top of the Barangay's base implementation.
    """
    def __init__(self, unique_id, model, quarterly_budget):
        super().__init__(unique_id, model)
        self.municipal_budget = quarterly_budget
        self.iec_active = False
        
        # --- FIX: Add this line so the Barangay can log LGU rewards here ---
        self.total_lgu_incentives_distributed = 0 

    def step(self):
        # The Mayor makes a strategic decision at the start of every Quarter (90 days)
        if self.model.tick % 90 == 0:
            self.run_decision_logic()

    def run_decision_logic(self):
        # 1. AI MODE (HuDRL)
        if self.model.policy_mode == "HuDRL" and self.model.rl_agent is not None:
            state = self.model.get_state()
            raw_action, _ = self.model.rl_agent.predict(state, deterministic=True)
            
            # --- INCREASED MULTIPLIER TO FORCE POLARIZED BUDGETS ---
            # Changing 2.0 to 5.0 forces the AI to make extreme, concentrated choices 
            # instead of smearing the budget equally.
            amplified = np.exp(raw_action * 10.0)
            
            # --- THE GRADUATION RULE ---
            # Matches the Gym Environment: If a barangay is doing great (>= 70%), 
            # artificially crush its AI allocation so the AI spends money elsewhere.
            compliance_rates = state[0:7]
            for i in range(7):
                if compliance_rates[i] >= 0.70:
                    amplified[i*3 : i*3+3] *= 0.01
                    
            total_desire = np.sum(amplified)
            
            if total_desire > 0:
                action_vector = amplified / total_desire
            else:
                action_vector = np.ones(21) / 21.0
            
        # 2. MANUAL STRATEGIES (Status Quo, Pure Enf, Pure Inc)
        else:
            action_vector = []
            share = 1.0 # Will distribute equal weights to all 7 barangays
            
            for b in self.model.barangays:
                if self.model.policy_mode == "pure_enforcement":
                    action_vector.extend([0.0, share, 0.0])
                elif self.model.policy_mode == "pure_incentives":
                    action_vector.extend([0.0, 0.0, share])
                else: 
                    # Default: "status_quo" -> All to IEC
                    action_vector.extend([share, 0.0, 0.0])
            
            # Normalize vector
            action_vector = np.array(action_vector)
            total_desire = np.sum(action_vector)
            if total_desire > 0:
                action_vector = action_vector / total_desire

        # 3. Transform the action_vector into Direct Interventions
        self.execute_intervention(action_vector)
        
        # Log the decisions to CSV
        self.model.log_quarterly_report(self.model.quarter)

    def execute_intervention(self, action_vector):
        """
        Directly implements LGU-funded levers across the 7 barangays.
        """
        import numpy as np

        # Calculate Global Compliance EARLY so we can use it for Phase-Shift AND Political Healing
        households = [a for a in self.model.schedule.agents if hasattr(a, 'is_compliant')]
        compliant_count = sum(1 for h in households if h.is_compliant)
        global_compliance = compliant_count / max(1, len(households))

        # --- ONLY APPLY HEURISTICS IF IN HuDRL MODE ---
        if self.model.policy_mode == "HuDRL":
            # =================================================================
            # 0. THE GRADUATION RULE (SUPPRESSION)
            # Kill funding for barangays that have hit the 'Habit Shield' threshold.
            # =================================================================
            current_compliances = [b.get_local_compliance() for b in self.model.barangays]
            for i in range(7):
                if current_compliances[i] >= 0.70:
                    # Artificially crush the allocation for successful barangays
                    action_vector[i*3 : i*3+3] *= 0.01
            
            # Re-normalize immediately so the subsequent heuristics work with 
            # the 'true' remaining budget.
            total_remaining = np.sum(action_vector)
            if total_remaining > 0:
                action_vector[:] = action_vector / total_remaining
                
            # =================================================================
            # 1. 'TARGET LOCK' HEURISTIC (THE SWAP)
            # =================================================================
            compliances = [b.get_local_compliance() for b in self.model.barangays]
            lowest_idx = np.argmin(compliances)
            
            # Calculate what the AI *intended* to give each barangay
            bgy_totals = [sum(action_vector[i*3:(i*3)+3]) for i in range(7)]
            intended_max_idx = np.argmax(bgy_totals)
            
            # If the AI aimed its biggest budget at the wrong barangay, redirect it!
            if lowest_idx != intended_max_idx:
                biggest_weapon = action_vector[intended_max_idx*3 : intended_max_idx*3+3].copy()
                smaller_allocation = action_vector[lowest_idx*3 : lowest_idx*3+3].copy()
                
                # SWAP THEM (Modifying in-place)
                action_vector[lowest_idx*3 : lowest_idx*3+3] = biggest_weapon
                action_vector[intended_max_idx*3 : intended_max_idx*3+3] = smaller_allocation
                
            # Re-normalize just to be mathematically safe (USING [:] FOR IN-PLACE)
            total_desire = np.sum(action_vector)
            if total_desire > 0:
                action_vector[:] = action_vector / total_desire

            # =================================================================
            # 2. THE DYNAMIC FINANCIAL GUARDRAIL (PHASE-SHIFT LOGIC)
            # Ensures the worst barangay receives AT LEAST 50% ONLY IF in Crisis.
            # =================================================================
            target_levers = action_vector[lowest_idx*3 : lowest_idx*3+3].copy()
            target_sum = np.sum(target_levers)
            
            # PHASE 1: CRISIS MODE (Force 50% Minimum)
            if global_compliance < 0.70:
                if target_sum < 0.50:
                    # Scale target up to exactly 50%
                    if target_sum > 0:
                        action_vector[lowest_idx*3 : lowest_idx*3+3] = (target_levers / target_sum) * 0.50
                    else:
                        action_vector[lowest_idx*3 : lowest_idx*3+3] = np.array([0.50/3, 0.50/3, 0.50/3])
                        
                    # Scale the remaining 6 barangays down to share the remaining 50%
                    other_indices = [i for i in range(21) if i not in [lowest_idx*3, lowest_idx*3+1, lowest_idx*3+2]]
                    other_levers = action_vector[other_indices].copy()
                    other_sum = np.sum(other_levers)
                    
                    if other_sum > 0:
                        action_vector[other_indices] = (other_levers / other_sum) * 0.50
                    else:
                        action_vector[other_indices] = 0.50 / 18.0
                        
                    # Re-normalize one final time to eliminate floating point errors (IN-PLACE)
                    action_vector[:] = action_vector / np.sum(action_vector)
                    
            # PHASE 2: MAINTENANCE MODE (Guardrail Removed)
            else:
                # The AI is allowed to naturally spread the budget however it wants
                # because the 70% threshold has been crossed!
                pass 
        # =================================================================

        # =====================================================================
        # CORE EXECUTION: THIS NOW RUNS FOR BOTH Vanilla PPO AND HuDRL
        # =====================================================================
        scale_factor = self.municipal_budget # Total LGU money to spend this quarter
        
        # --- Initialize tracker for Political Recovery ---
        total_incentives_spent = 0.0 
        
        for i, bgy in enumerate(self.model.barangays):
            idx = i * 3
            # Extract LGU-specific budget for this specific Barangay
            lgu_iec = action_vector[idx] * scale_factor
            lgu_enf = action_vector[idx+1] * scale_factor
            lgu_inc = action_vector[idx+2] * scale_factor
            
            # Save these for the CSV Logger to read
            bgy.lgu_iec_fund = lgu_iec
            bgy.lgu_enf_fund = lgu_enf
            bgy.lgu_inc_fund = lgu_inc

            # --- LEVER 1: PURE ENFORCEMENT (Municipal Inspectors) ---
            self.deploy_municipal_inspectors(bgy, lgu_enf)

            # --- LEVER 2: UNIVERSAL INCENTIVES (LGU Reward Pool) ---
            bgy.lgu_incentive_fund = lgu_inc
            total_incentives_spent += lgu_inc  # <--- TRACK IT HERE!

            # --- LEVER 3: UNIFIED IEC (Municipal Awareness) ---
            if lgu_iec > 0:
                self.run_municipal_iec(bgy, lgu_iec)

        # ==========================================================
        # POLITICAL CAPITAL RECOVERY (THE APPROVAL RATING)
        # ==========================================================
        
        # 1. Every 5,000 PHP spent on Incentives buys back 0.01 Political Capital
        ayuda_boost = (total_incentives_spent / 5000.0) * 0.01
        
        # 2. Tiered "Clean City" Bonus (Using the global_compliance from line 10)
        clean_city_boost = 0.0
        if global_compliance >= 0.85:
            clean_city_boost = 0.20   # BUFFED: The Mayor is a hero, massive healing prevents late-game crash
        elif global_compliance >= 0.70:
            clean_city_boost = 0.10   
        elif global_compliance >= 0.40:
            clean_city_boost = 0.02   
        
        # 3. Apply the healing (Capped at 1.0 or 100% approval)
        self.model.political_capital = min(1.0, self.model.political_capital + ayuda_boost + clean_city_boost)
        
    def deploy_municipal_inspectors(self, bgy, fund):
        import random # Ensure this is at the top of your file if not already there

        # 400 PHP * 30 Days = 12,000 PHP per enforcer task force contract
        target_inspectors = int(fund // (400 * 30)) 
        
        current_municipal = [a for a in self.model.schedule.agents 
                             if isinstance(a, EnforcementAgent) 
                             and getattr(a, 'is_municipal', False) 
                             and a.barangay_id == bgy.unique_id]

        current_count = len(current_municipal)

        # Add new inspectors if funding allows
        if current_count < target_inspectors:
            inspectors_to_add = target_inspectors - current_count
            for _ in range(inspectors_to_add):
                # IMPROVED UNIQUE ID: Include a random salt to prevent collisions
                new_id = f"LGU_ENF_{bgy.unique_id}_{self.model.tick}_{random.randint(10000, 99999)}"
                
                # --- SAFETY CHECK ---
                if new_id in self.model.schedule._agents:
                    continue # Skip if ID miraculously exists
                
                # Instantiate the enforcer
                inspector = EnforcementAgent(new_id, self.model, bgy.unique_id)
                
                # Configure their specific Task Force attributes
                inspector.is_municipal = True
                inspector.fine_amount = 1000 
                inspector.contract_days = 30  # <--- THE 30-DAY EXPIRATION TIMER
                
                # Position them on the grid
                pos = (self.random.randrange(self.model.grid.width), self.random.randrange(self.model.grid.height))
                
                self.model.schedule.add(inspector)
                self.model.grid.place_agent(inspector, pos)
                
        # Remove excess inspectors if budget was slashed 
        # (Though most will naturally expire after 30 days anyway due to the timer)
        elif current_count > target_inspectors:
            inspectors_to_remove = current_count - target_inspectors
            for inspector in current_municipal[:inspectors_to_remove]:
                self.model.grid.remove_agent(inspector)
                self.model.schedule.remove(inspector)

    def run_municipal_iec(self, bgy, fund):
        # --- LGU MACRO-IEC EQUATION ---
        # Based on MENRO Interview: Uses Radio Broadcasts (101.3 FM) and Large Assemblies
        C_RADIO = 500.0
        C_EVENT = 5000.0
        TARGET_RADIO = 90 # Goal: Daily radio spots
        TARGET_EVENTS = 3 # Goal: Monthly large assemblies

        # 1. Buy Radio Airtime first
        n_radio = min(TARGET_RADIO, int(fund // C_RADIO))
        rem = max(0.0, fund - (n_radio * C_RADIO))
        
        # 2. Fund Large Events with remainder
        n_events = min(TARGET_EVENTS, int(rem // C_EVENT))

        # Calculate LGU-driven intensity (How strong the campaign is this quarter)
        lgu_iec_intensity = min(1.0, (n_radio / TARGET_RADIO) * 0.4 + (n_events / TARGET_EVENTS) * 0.6)

        # Apply direct impact: LGU campaigns boost 'Attitude' and 'Social Norms' 
        households = [a for a in self.model.schedule.agents 
                      if type(a).__name__ == "HouseholdAgent" and getattr(a, 'barangay_id', None) == bgy.unique_id]
        
        if households and lgu_iec_intensity > 0:
            # BUFFED: Increased impact from 0.05 to 0.15 so the AI isn't wasting its money
            impact = lgu_iec_intensity * 0.15 
            
            import random # Ensure random is available for the trigger
            for h in households:
                h.attitude = min(1.0, h.attitude + impact)
                h.sn = min(1.0, h.sn + (impact * 0.5))
                
                # --- THE REALISTIC CONVERSION TRIGGER ---
                # They must be highly convinced (>75%), and even then, only 20% actually change habits.
                if h.attitude > 0.75:  
                    if random.random() < 0.20:
                        h.is_compliant = True