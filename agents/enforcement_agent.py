import mesa
import math
import random
from agents.household_agent import HouseholdAgent

class EnforcementAgent(mesa.Agent):
    def __init__(self, unique_id, model, barangay_id, patrol_range=10):
        super().__init__(unique_id, model)
        self.barangay_id = barangay_id 
        self.patrol_range = patrol_range
        self.fine_amount = 500
        self.is_municipal = False 
        
        # --- THE MISSING LINE ---
        self.visited_households = set()

    def step(self):
        import random
        
        # ==========================================================
        # 1. 30-DAY CONTRACT CHECK & SAFE REMOVAL
        # ==========================================================
        if getattr(self, 'is_municipal', False):
            if hasattr(self, 'contract_days'):
                self.contract_days -= 1
                if self.contract_days <= 0:
                    # CRITICAL FIX: Only remove from grid if the agent has a position
                    if self.pos is not None:
                        self.model.grid.remove_agent(self)
                    
                    # Only remove from schedule if it hasn't been removed already
                    if self in self.model.schedule.agents:
                        self.model.schedule.remove(self)
                    return # Exit immediately so no further logic runs

        self.visited_households.clear() 
        is_municipal = getattr(self, 'is_municipal', False)
        
        # ==========================================================
        # 2. STOCHASTIC PATROLLING (The Nerf)
        # ==========================================================
        if is_municipal:
            max_daily_capacity = 25 
        else:
            max_daily_capacity = 1
            if random.random() > 0.10: 
                return # Skips patrol for today

        # Enforcement Logic
        bgy_households = self.model.households_by_bgy.get(self.barangay_id, [])
        targets = [h for h in bgy_households if not h.is_compliant]
        
        if targets:
            caught_count = min(max_daily_capacity, len(targets))
            caught = random.sample(targets, caught_count)
            
            for target in caught:
                if hasattr(target, 'get_fined'):
                    target.get_fined(self.fine_amount)

        # ==========================================================
        # 3. THE VISUALS: Safe Movement
        # ==========================================================
        if not getattr(self.model, 'train_mode', False):
            # ADDITIONAL SAFETY: move_agent will also crash if pos is None
            if self.pos is not None:
                possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
                if possible_steps:
                    self.model.grid.move_agent(self, getattr(self.model, 'random', random).choice(possible_steps))