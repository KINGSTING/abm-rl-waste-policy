import mesa
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import random
import os
import csv 
from stable_baselines3 import PPO

import barangay_config as config
from agents.household_agent import HouseholdAgent
from agents.barangay_agent import BarangayAgent
from agents.enforcement_agent import EnforcementAgent
from agents.mayor_agent import MayorAgent 

def compute_global_compliance(model):
    agents = [a for a in model.schedule.agents if isinstance(a, HouseholdAgent)]
    if not agents: return 0.0
    return sum(1 for a in agents if a.is_compliant) / len(agents)

class BacolodModel(mesa.Model):
    # =================================================================
    # THE FIX: ADDED active_barangay TO THE ORIGINAL __init__
    # =================================================================
    def __init__(self, seed=None, train_mode=False, policy_mode="HuDRL", 
                 active_barangay="BGY_0", behavior_override=None, current_view="BGY_0",
                 sens_t_low=None, sens_t_high=None): # ADDED SENSITIVITY ARGS
        if seed is not None:
            super().__init__(seed=seed)
            self._seed = seed
            np.random.seed(seed)
            random.seed(seed)
        else:
            super().__init__()

        # 1. CORE MESA INFRASTRUCTURE (Define FIRST)
        self.grid_width = 50   
        self.grid_height = 50 
        self.grid = MultiGrid(self.grid_width, self.grid_height, torus=False)
        self.schedule = RandomActivation(self)
        self.running = True

        # 2. MODEL CLOCKS & VARIABLES
        self.tick = 0       
        self.quarter = 1    
        self.train_mode = train_mode
        self.behavior_override = behavior_override
        
        # THE FIX: Save the sensitivity arguments as instance attributes
        self.sens_t_low = sens_t_low
        self.sens_t_high = sens_t_high
        
        # These are what the HouseholdAgent looks for
        self.threshold_low = sens_t_low if sens_t_low is not None else 0.40
        self.threshold_high = sens_t_high if sens_t_high is not None else 0.70

        # Policy String Sanitization
        raw_policy = str(policy_mode).strip().lower()
        if "enforcement" in raw_policy: self.policy_mode = "pure_enforcement"
        elif "incentive" in raw_policy: self.policy_mode = "pure_incentives"
        elif "hudrl" in raw_policy or "mayor" in raw_policy: self.policy_mode = "HuDRL"
        else: self.policy_mode = "status_quo"

        # 3. CSV LOGGING SETUP
        results_dir = "results"
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        self.csv_local = os.path.join(results_dir, f"{self.policy_mode}_1_LOCAL_BASE.csv")
        self.csv_mayor = os.path.join(results_dir, f"{self.policy_mode}_2_MAYOR_INTERVENTION.csv")
        self.csv_global = os.path.join(results_dir, f"{self.policy_mode}_3_GLOBAL_SUMMARY.csv")
        
        # Only write headers if we aren't in a parallel test
        if not self.behavior_override and not self.train_mode:
            self._initialize_csv_headers()

        # 4. LOAD RL MODEL (FIXED: Allow loading during sensitivity tests)
        if self.policy_mode == "HuDRL":
            model_path = "models/ppo/bacolod_ppo_final.zip"
            if os.path.exists(model_path):
                # Using CPU for multiprocessing stability
                self.rl_agent = PPO.load(model_path, device="cpu")
                if not self.train_mode: print(f"HuDRL Brain Active.")

        # Budget & Political Capital Params
        self.annual_budget = config.ANNUAL_BUDGET
        self.current_budget = self.annual_budget
        self.quarterly_budget = self.annual_budget / 4 
        self.total_fines_collected = 0
        self.total_incentives_distributed = 0
        self.total_enforcement_cost = 0
        self.total_iec_cost = 0
        self.recent_fines_collected = 0
        self.political_capital = 1.0     
        self.alpha_sensitivity = 0.0010 
        self.beta_recovery = 0.0002      

        # 5. AGENT CREATION
        self.barangays = []
        self.agent_id_counter = 0
        self.households_by_bgy = {}
        
        for i, b_conf in enumerate(config.BARANGAY_LIST):
            b_agent = BarangayAgent(f"BGY_{i}", self, local_budget=b_conf["local_budget"])
            b_agent.name = b_conf["name"]
            b_agent.n_households = b_conf["N_HOUSEHOLDS"]
            if "allocation_profile" in b_conf:
                pk = b_conf["allocation_profile"]
                b_agent.local_allocation_ratios = config.ALLOCATION_PROFILES.get(pk, config.ALLOCATION_PROFILES["Ezperanza"])
            self.schedule.add(b_agent)
            self.barangays.append(b_agent)
            
            # Behavior Data + Sensitivity Injection
            profile_key = b_conf.get("behavior_profile", "Poblacion") 
            behavior_data = config.BEHAVIOR_PROFILES.get(profile_key, config.BEHAVIOR_PROFILES["Poblacion"]).copy()
            
            # Now self.sens_t_low and self.sens_t_high exist!
            if self.sens_t_low is not None: 
                behavior_data["threshold_low"] = self.sens_t_low
            if self.sens_t_high is not None: 
                behavior_data["threshold_high"] = self.sens_t_high

            income_probs = list(config.INCOME_PROFILES[b_conf["income_profile"]])
            for _ in range(b_conf["N_HOUSEHOLDS"]):
                x = self.random.randrange(self.grid_width)
                y = self.random.randrange(self.grid_height)
                income = np.random.choice([1, 2, 3], p=income_probs)
                is_compliant = (random.random() < b_conf["initial_compliance"])
                a = HouseholdAgent(self.agent_id_counter, self, income, is_compliant, behavior_data)
                self.agent_id_counter += 1
                a.barangay = b_agent
                a.barangay_id = b_agent.unique_id
                if b_agent.unique_id not in self.households_by_bgy: self.households_by_bgy[b_agent.unique_id] = []
                self.households_by_bgy[b_agent.unique_id].append(a)
                self.schedule.add(a)
                self.grid.place_agent(a, (x, y))

        # 6. MAYOR & PRE-FILTERED LISTS
        self.mayor = MayorAgent("MAYOR_0", self, self.quarterly_budget)
        self.schedule.add(self.mayor)
        self.mayor.run_decision_logic() 

        # OPTIMIZATION: Create static lists so we don't loop over 'schedule.agents' every step
        self.household_list = [a for a in self.schedule.agents if isinstance(a, HouseholdAgent)]
        self.decision_list = [a for a in self.schedule.agents if isinstance(a, (HouseholdAgent, EnforcementAgent, MayorAgent))]

        self.datacollector = DataCollector(model_reporters={
            "Global Compliance": lambda m: compute_global_compliance(m) * 100.0,
            "Total Fines": lambda m: m.total_fines_collected,
            "Political Capital": lambda m: m.political_capital
        })
        self.datacollector.collect(self)

    def update_political_capital(self):
        avg_enforcement = 0
        if self.barangays:
            avg_enforcement = sum(b.enforcement_intensity for b in self.barangays) / len(self.barangays)
            
        all_households = [a for a in self.schedule.agents if isinstance(a, HouseholdAgent)]
        if all_households:
            avg_attitude = np.mean([a.attitude for a in all_households])
        else:
            avg_attitude = 0.5 
            
        attitude_modifier = 2.0 * (1.0 - avg_attitude)
        effective_decay = (self.alpha_sensitivity * avg_enforcement) * attitude_modifier
        recovery = self.beta_recovery * (1.0 - avg_enforcement)
        self.political_capital = max(0.0, min(1.0, self.political_capital - effective_decay + recovery))

    def calculate_costs(self):
        total_iec_alloc = sum(b.iec_fund for b in self.barangays)
        total_enf_alloc = sum(b.enf_fund for b in self.barangays)
        daily_fixed_cost = (total_iec_alloc + total_enf_alloc) / 90.0
        self.total_enforcement_cost += (total_enf_alloc / 90.0)
        self.total_iec_cost += (total_iec_alloc / 90.0)
        self.current_budget = self.current_budget - daily_fixed_cost + self.recent_fines_collected
        self.recent_fines_collected = 0

    def adjust_enforcement_agents(self, barangay_agent):
        existing_agents = [a for a in self.schedule.agents 
                        if isinstance(a, EnforcementAgent) 
                        and a.barangay_id == barangay_agent.unique_id
                        and not getattr(a, 'is_municipal', False)]
        
        current_count = len(existing_agents)
        target_count = barangay_agent.n_enforcers 
        
        if current_count < target_count:
            for i in range(target_count - current_count):
                new_id = f"Tanod_{barangay_agent.unique_id}_{self.tick}_{i}"
                pos = (self.random.randrange(self.grid.width), self.random.randrange(self.grid.height))
                agent = EnforcementAgent(new_id, self, barangay_agent.unique_id)
                self.schedule.add(agent)
                self.grid.place_agent(agent, pos)
                
        elif current_count > target_count:
            for i in range(current_count - target_count):
                agent_to_remove = existing_agents[i]
                self.grid.remove_agent(agent_to_remove)
                self.schedule.remove(agent_to_remove)

    def log_quarterly_report(self, quarter):
        if self.behavior_override: return
        
        all_attitudes = []

        with open(self.csv_local, mode='a', newline='') as file_local, \
             open(self.csv_mayor, mode='a', newline='') as file_mayor:
            
            writer_local = csv.writer(file_local)
            writer_mayor = csv.writer(file_mayor)

            for b in self.barangays:
                local_total = b.local_quarterly_budget
                l_iec_pct = (b.iec_fund / local_total * 100) if local_total > 0 else 0
                l_enf_pct = (b.enf_fund / local_total * 100) if local_total > 0 else 0
                l_inc_pct = (b.inc_fund / local_total * 100) if local_total > 0 else 0
                
                local_tanods = len([a for a in self.schedule.agents 
                                    if isinstance(a, EnforcementAgent) 
                                    and a.barangay_id == b.unique_id 
                                    and not a.is_municipal])
                
                writer_local.writerow([
                    quarter, b.name, f"{local_total:.2f}",
                    f"{b.iec_fund:.2f}", f"{l_iec_pct:.1f}%", 
                    f"{b.enf_fund:.2f}", f"{l_enf_pct:.1f}%", 
                    f"{b.inc_fund:.2f}", f"{l_inc_pct:.1f}%",
                    local_tanods, f"{b.get_local_compliance():.2%}"
                ])

                lgu_iec = getattr(b, 'lgu_iec_fund', 0)
                lgu_enf = getattr(b, 'lgu_enf_fund', 0)
                lgu_inc = getattr(b, 'lgu_incentive_fund', 0)
                lgu_total = lgu_iec + lgu_enf + lgu_inc
                
                m_iec_pct = (lgu_iec / lgu_total * 100) if lgu_total > 0 else 0
                m_enf_pct = (lgu_enf / lgu_total * 100) if lgu_total > 0 else 0
                m_inc_pct = (lgu_inc / lgu_total * 100) if lgu_total > 0 else 0
                
                m_share_overall = (lgu_total / self.quarterly_budget * 100) if self.quarterly_budget > 0 else 0
                
                lgu_inspectors = len([a for a in self.schedule.agents 
                                      if isinstance(a, EnforcementAgent) 
                                      and a.barangay_id == b.unique_id 
                                      and a.is_municipal])

                writer_mayor.writerow([
                    quarter, b.name, f"{lgu_total:.2f}", f"{m_share_overall:.1f}%",
                    f"{lgu_iec:.2f}", f"{m_iec_pct:.1f}%", 
                    f"{lgu_enf:.2f}", f"{m_enf_pct:.1f}%", 
                    f"{lgu_inc:.2f}", f"{m_inc_pct:.1f}%",
                    lgu_inspectors
                ])

                households = self.households_by_bgy.get(b.unique_id, [])
                if households:
                    all_attitudes.extend([a.attitude for a in households])

        with open(self.csv_global, mode='a', newline='') as file_global:
            writer_global = csv.writer(file_global)
            global_comp = compute_global_compliance(self)
            avg_att = np.mean(all_attitudes) if all_attitudes else 0.0
            
            writer_global.writerow([
                quarter, f"{global_comp:.2%}", f"{self.political_capital:.4f}", 
                self.total_fines_collected, f"{avg_att:.3f}"
            ])

        if not self.train_mode:
            print(f" > Split Reports (Local, Mayor, Global) saved for Quarter {quarter}")

    def step(self):
        self.tick += 1
        self.schedule.steps += 1
        self.schedule.time += 1

        # 1. Quarterly logic
        if self.tick % 90 == 0:
            self.quarter += 1
            # We still use household_list here because Households are never 'deleted'
            for a in self.household_list:
                a.redeemed_this_quarter = False
            
            if not self.train_mode and not self.behavior_override:
                print(f" >> New Quarter: {self.quarter}")

        # 2. Step Barangays
        for b in self.barangays: 
            b.step()
            
        # 3. THE DYNAMIC FIX: Loop through the actual schedule
        # This ensures Municipal Inspectors actually 'act'
        for a in self.schedule.agents:
            if isinstance(a, (HouseholdAgent, EnforcementAgent, MayorAgent)):
                a.step()

        self.update_political_capital() 
        self.calculate_costs()

        if not self.train_mode:
            self.datacollector.collect(self)

        # 12 Quarters = 1080 ticks
        if self.tick >= 1080: 
            self.running = False
             
    def get_state(self):
        compliance_rates = [b.get_local_compliance() for b in self.barangays]
        attitude_rates = []
        
        for b in self.barangays:
            households = self.households_by_bgy.get(b.unique_id, [])
            avg_att = np.mean([a.attitude for a in households]) if households else 0.0
            attitude_rates.append(avg_att)

        norm_budget = max(0.0, min(1.0, self.current_budget / self.annual_budget))
        norm_time = max(0.0, min(1.0, self.quarter / 12.0))
        p_cap = max(0.0, min(1.0, self.political_capital)) 
        
        state = compliance_rates + attitude_rates + [norm_budget, norm_time, p_cap]
        return np.array(state, dtype=np.float32)
    
    # INDENT THIS BLOCK BY 4 SPACES
    def _initialize_csv_headers(self):
        """Initializes the CSV files with their respective header rows."""
        import csv
        
        # 1. Local Base Headers
        with open(self.csv_local, mode='w', newline='') as file_local:
            writer_local = csv.writer(file_local)
            writer_local.writerow([
                "Quarter", "Barangay", "Local Budget", 
                "IEC Fund", "IEC %", "ENF Fund", "ENF %", 
                "INC Fund", "INC %", "Local Tanods", "Compliance Rate"
            ])

        # 2. Mayor Intervention Headers
        with open(self.csv_mayor, mode='w', newline='') as file_mayor:
            writer_mayor = csv.writer(file_mayor)
            writer_mayor.writerow([
                "Quarter", "Barangay", "LGU Total", "LGU Share %",
                "LGU IEC", "LGU IEC %", "LGU ENF", "LGU ENF %", 
                "LGU INC", "LGU INC %", "LGU Inspectors"
            ])

        # 3. Global Summary Headers
        with open(self.csv_global, mode='w', newline='') as file_global:
            writer_global = csv.writer(file_global)
            writer_global.writerow([
                "Quarter", "Global Compliance", "Political Capital", 
                "Total Fines Collected", "Average Attitude"
            ])