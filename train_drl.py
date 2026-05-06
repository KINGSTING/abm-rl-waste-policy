import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from bacolod_gym import BacolodGymEnv

def make_env(rank, seed=0):
    def _init():
        return BacolodGymEnv()
    return _init

def train():
    num_cpu = 4 
    
    # WE ARE BACK TO 4 CORES!
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env) 

    # Increase timesteps to at least 500k - 1 Million for a 21D space
    TIMESTEPS = 50000
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003, 
        n_steps=1024,        # Increased from 60 to gather a full trajectory before updating
        batch_size=128,      # Increased from 60 for more stable gradient updates
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,       # Keeps exploration healthy
        tensorboard_log="./logs/",
        device="cpu"
    )

    print(f"Bacolod DRL: Fast-Track Training Started on {num_cpu} Cores...")
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    
    model.save("models/ppo/bacolod_ppo_no_jackpot")
    print("\n[SUCCESS] Training Complete! Ready for your comparison.")

if __name__ == "__main__":
    train()