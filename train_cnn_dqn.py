import jax.numpy as jnp
import jax.random as jrandom
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generals import GeneralsEnv, get_observation
from generals.agents import ExpanderAgent
from cnn_dqn_agent import CNNDQNAgent

GRID_DIMS  = (8, 8)
TRUNCATION = 400
SAVE_PATH  = "trained_cnn_dqn.pkl"
SAVE_EVERY = 50

def train():
    env     = GeneralsEnv(grid_dims=GRID_DIMS, truncation=TRUNCATION)
    agent_0 = ExpanderAgent()
    agent_1 = CNNDQNAgent(learning_rate=0.0001)

    start_episode = 0
    total_wins    = 0

    meta_path = SAVE_PATH + ".meta"
    agent_1.load(SAVE_PATH)
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        start_episode = meta["episode"]
        total_wins    = meta["total_wins"]
        print(f"📂 Resuming from episode {start_episode}  (wins so far: {total_wins})\n")
    else:
        print("✓ Starting CNN-DQN training (256-action spatial agent)\n")

    episode = start_episode

    while True:
        episode += 1
        key   = jrandom.PRNGKey(episode)
        state = env.reset(key)

        terminated = truncated = False
        prev_state = None
        prev_obs_1 = None
        episode_losses = []

        while not (terminated or truncated):
            obs_0 = get_observation(state, 0)
            obs_1 = get_observation(state, 1)

            key, k1, k2 = jrandom.split(key, 3)
            action_0 = agent_0.act(obs_0, k1)
            action_1 = agent_1.act(obs_1, k2)
            actions  = jnp.stack([action_0, action_1])

            timestep, next_state = env.step(state, actions)

            if prev_state is not None and prev_obs_1 is not None:
                prev_enc = agent_1.encode_state(prev_obs_1)
                curr_enc = agent_1.encode_state(obs_1)
                reward   = agent_1.get_reward_from_states(prev_state, state, 1)
                loss     = agent_1.update(prev_enc, curr_enc, reward, bool(timestep.terminated))
                if loss > 0:
                    episode_losses.append(loss)

            prev_state = state
            prev_obs_1 = obs_1
            state      = next_state

            terminated = bool(timestep.terminated)
            truncated  = bool(timestep.truncated)

        agent_1.end_episode()

        winner_id = int(timestep.info.winner)
        if winner_id == 1:
            total_wins += 1
            print(f"🎉 Ep {episode}: WON! W:{total_wins} ε:{agent_1.epsilon:.3f} Buf:{len(agent_1.replay_buffer)}")
        elif episode % 10 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            print(f"Ep {episode}: Lost | W:{total_wins} ε:{agent_1.epsilon:.3f} L:{avg_loss:.3f} Buf:{len(agent_1.replay_buffer)}")

        if episode % 100 == 0:
            win_rate = total_wins / episode   # true overall win rate
            print(f"\n{'='*50}")
            print(f"Episode {episode}: Win rate {win_rate:.1%}  (total wins: {total_wins})")
            print(f"{'='*50}\n")

        if episode % SAVE_EVERY == 0:
            agent_1.save(SAVE_PATH)
            with open(meta_path, "wb") as f:
                pickle.dump({"episode": episode, "total_wins": total_wins}, f)

if __name__ == "__main__":
    train()