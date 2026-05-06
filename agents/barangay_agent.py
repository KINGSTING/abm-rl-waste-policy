import mesa
import random
import math
from agents.household_agent import HouseholdAgent
from agents.enforcement_agent import EnforcementAgent
import barangay_config as config

class BarangayAgent(mesa.Agent):
    def __init__(self, unique_id, model, local_budget=0):
        super().__init__(unique_id, model)
        
        # --- 1. Identity, Demographics & Config Extraction ---
        try:
            # Extract index from ID (e.g., "BGY_0" -> 0)
            idx = int(str(unique_id).split('_')[1])
            b_conf = config.BARANGAY_LIST[idx]
            self.name = b_conf["name"]
            self.n_households = b_conf["N_HOUSEHOLDS"]
            
            # Pull the exact allocation profile assigned to this barangay
            profile_key = b_conf.get("allocation_profile", "Poblacion")
            self.local_allocation_ratios = config.ALLOCATION_PROFILES.get(profile_key, {"enf": 0.33, "inc": 0.33, "iec": 0.34})
        except:
            self.name = ""
            self.n_households = 1  
            self.local_allocation_ratios = {"enf": 0.33, "inc": 0.33, "iec": 0.34}
            
        # --- 2. State Metrics ---
        self.compliance_rate = 0.0
        self.total_households = 0
        self.compliant_count = 0
        
        # --- 3. Policy Variables (LOCAL TOTALS) ---
        self.fine_amount = 500
        
        # LGU Intervention Buckets (Managed by MayorAgent)
        self.lgu_iec_fund = 0.0
        self.lgu_enf_fund = 0.0
        self.lgu_incentive_fund = 0.0
        
        # --- 4. Intensities & Outcomes ---
        self.enforcement_intensity = 0.5
        self.iec_intensity = 0.0
        self.incentive_val = 500.0  # Prize amount per winner
        self.n_enforcers = 0

        # --- 5. Financials ---
        self.local_annual_budget = local_budget
        self.local_quarterly_budget = local_budget / 4.0 
        self.current_cash_on_hand = 0.0
        
        # Pre-calculate Quarter 1 funds so CSVs are immediately accurate
        self.enf_fund = self.local_quarterly_budget * self.local_allocation_ratios.get("enf", 0.0)
        self.inc_fund = self.local_quarterly_budget * self.local_allocation_ratios.get("inc", 0.0)
        self.iec_fund = self.local_quarterly_budget * self.local_allocation_ratios.get("iec", 0.0)

        # Initialize local policy immediately on Day 0
        self.setup_local_policy()

    def setup_local_policy(self):
        """Sets up the LOCAL implementation based purely on the Barangay's own IRA/Budget."""
        # --- A. Calculate Local Base Allocations ---
        self.enf_fund = self.local_quarterly_budget * self.local_allocation_ratios["enf"]
        self.inc_fund = self.local_quarterly_budget * self.local_allocation_ratios["inc"]
        self.iec_fund = self.local_quarterly_budget * self.local_allocation_ratios["iec"]

        # --- B. FULL IMPLEMENTATION: Cost of Enforcement (Eq. 3.1) ---
        cost_per_enforcer_quarterly = 400 * 66
        self.n_enforcers = int(self.enf_fund // cost_per_enforcer_quarterly)
        self.actual_enforcement_cost = self.n_enforcers * cost_per_enforcer_quarterly

        # Let the model know it needs to sync local Tanods
        self.model.adjust_enforcement_agents(self)

        # --- C. FULL IMPLEMENTATION: Cost of Barangay IEC (Micro-Level) ---
        # Based on interviews: Barangays rely on Recorida and Purok Meetings
        C_RECORIDA = 200.0  # Gas for barangay vehicle
        C_PUROK = 1000.0    # Snacks/materials for neighborhood meetings
        TARGET_RECORIDA = 30 # Goal: Every 3 days
        TARGET_PUROK = 3     # Goal: Once a month
        
        # 1. Fund Recoridas first (Cheapest, widest local reach)
        self.n_recorida = min(TARGET_RECORIDA, int(self.iec_fund // C_RECORIDA))
        rem = max(0.0, self.iec_fund - (self.n_recorida * C_RECORIDA))
        
        # 2. Fund Purok Meetings with whatever is left
        self.n_purok = min(TARGET_PUROK, int(rem // C_PUROK))
        self.actual_iec_cost = (self.n_recorida * C_RECORIDA) + (self.n_purok * C_PUROK)
        
        # Calculate local intensity (0.0 to 1.0)
        self.iec_intensity = min(1.0, (self.n_recorida / TARGET_RECORIDA) * 0.5 + (self.n_purok / TARGET_PUROK) * 0.5)

        # --- D. Operational Cash for Incentives (Eq. 3.2) ---
        self.current_cash_on_hand = self.inc_fund

    def local_iec_implementation(self):
        """Provides a small daily boost to local households representing localized IEC (Recorida/Purok)"""
        if self.iec_intensity > 0:
            impact = self.iec_intensity * 0.0005 # Small daily drip of awareness
            households = self.model.households_by_bgy.get(self.unique_id, [])
            for h in households:
                h.attitude = min(0.95, h.attitude + impact)

    def distribute_local_incentives(self):
        """Logic for the 'Selected Few' Draw at end of Quarter using LOCAL funds."""
        max_winners = int(self.current_cash_on_hand // 500)
        
        # High-Speed Fetch
        households = self.model.households_by_bgy.get(self.unique_id, [])
        eligible_households = [h for h in households if h.is_compliant]
        
        if not eligible_households or max_winners == 0:
            return

        num_to_pay = min(len(eligible_households), max_winners)
        winners = random.sample(eligible_households, num_to_pay)
        
        for household in winners:
            household.receive_incentive(500)
            self.current_cash_on_hand -= 500
            
        losers = [h for h in eligible_households if h not in winners]
        for household in losers:
            household.attitude -= 0.15 
            household.perceived_unfairness = True

    def get_local_compliance(self):
        """High-speed compliance calculation"""
        households = self.model.households_by_bgy.get(self.unique_id, [])
        self.total_households = len(households)
        
        if self.total_households > 0:
            self.compliant_count = sum(1 for h in households if h.is_compliant)
            self.compliance_rate = self.compliant_count / self.total_households
        else:
            self.compliance_rate = 0.0
            
        return self.compliance_rate

    def step(self):
        self.get_local_compliance()
        self.local_iec_implementation()
        
        # Calculate final enforcement pressure (Local Tanods + LGU Intervention)
        self.update_enforcement_intensity() 
        
        # End of Quarter Actions
        if self.model.tick > 0 and self.model.tick % 90 == 0:
            self.distribute_local_incentives()
            self.setup_local_policy() # Refresh funds for the next quarter

    def give_reward(self, amount):
        """
        Bridge function to satisfy the HouseholdAgent's request for a reward.
        Layers the LGU Universal Reward pool on top of the Local pool.
        """
        # 1. Try to pay from Local Barangay Grocery Fund
        if self.current_cash_on_hand >= amount:
            self.current_cash_on_hand -= amount
            return True
            
        # 2. If Barangay is broke, check if the Mayor allocated Universal Incentives here
        elif self.lgu_incentive_fund >= amount:
            self.lgu_incentive_fund -= amount
            self.model.mayor.total_lgu_incentives_distributed += amount 
            return True
            
        # 3. No funds available in either tier
        return False
        
    def update_enforcement_intensity(self):
        # 1. Find currently active Tanods
        active_tanods = [a for a in self.model.schedule.agents 
                         if type(a).__name__ == "EnforcementAgent" and getattr(a, 'barangay_id', None) == self.unique_id and not getattr(a, 'is_municipal', False)]
        
        # FIX: Add a 0.30 "Ambient Fear" so citizens never feel 100% invincible.
        target_ideal = max(1, self.n_households // 100) 
        base_fear = 0.30
        self.enforcement_intensity = base_fear + min(0.70, len(active_tanods) / target_ideal)
        
        # 2. If we have less tanods than the budget allows, spawn more
        if len(active_tanods) < self.n_enforcers:
                for i in range(self.n_enforcers - len(active_tanods)):
                    # Use your existing counter instead of next_id()
                    tanod = EnforcementAgent(self.model.agent_id_counter, self.model, self.unique_id)
                    self.model.agent_id_counter += 1
                    
                    tanod.is_municipal = False
                
                # Place them randomly at a household in their own barangay
                households = self.model.households_by_bgy.get(self.unique_id, [])
                if households:
                    pos = self.model.random.choice(households).pos
                    self.model.grid.place_agent(tanod, pos)
                    self.model.schedule.add(tanod)