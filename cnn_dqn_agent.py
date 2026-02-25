import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import random
from collections import deque
import functools
import pickle
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from generals import compute_valid_move_mask, get_observation

# Simple action space: 0=pass, 1=up, 2=down, 3=left, 4=right
NUM_ACTIONS = 5


class SpatialQNetwork(nn.Module):
    """
    CNN that takes full board (5, 8, 8) and outputs Q-values for 5 actions.
    """
    num_actions: int = 5

    @nn.compact
    def __call__(self, x):
        if x.ndim == 3:
            x = jnp.expand_dims(x, 0)  # (1, 5, 8, 8)

        x = jnp.transpose(x, (0, 2, 3, 1))  # (B, 8, 8, 5)

        # Convolutional layers - learn spatial patterns
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Dense layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)  # 5 Q-values

        return x


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = jnp.array([e[0] for e in batch])
        actions = jnp.array([e[1] for e in batch])
        rewards = jnp.array([e[2] for e in batch])
        next_states = jnp.array([e[3] for e in batch])
        dones = jnp.array([e[4] for e in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class CNNDQNAgent:
    def __init__(self, id="CNN_DQN_5", learning_rate=0.0001):
        self.id = id
        self.num_actions = NUM_ACTIONS
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # Faster than 256-action version
        self.gamma = 0.95
        self.step_counter = 0
        self.total_updates = 0

        self.network = SpatialQNetwork(num_actions=NUM_ACTIONS)

        key = jax.random.PRNGKey(42)
        dummy_state = jnp.zeros((5, 8, 8), dtype=jnp.float32)
        self.params = self.network.init(key, dummy_state)
        self.target_params = self.params

        self.target_update_freq = 100  # Update target network every 100 steps

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=learning_rate)
        )
        self.opt_state = self.optimizer.init(self.params)

        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.train_freq = 4

        self.last_state = None
        self.last_action = None

    # ── Save / Load ────────────────────────────────────────────────────────────

    def save(self, path):
        data = {
            "params": self.params,
            "target_params": self.target_params,
            "opt_state": self.opt_state,
            "epsilon": self.epsilon,
            "step_counter": self.step_counter,
            "total_updates": self.total_updates,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"💾 Saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            print(f"⚠️  No save at {path}, starting fresh.")
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.params = data["params"]
        self.target_params = data["target_params"]
        self.opt_state = data["opt_state"]
        self.epsilon = data["epsilon"]
        self.step_counter = data["step_counter"]
        self.total_updates = data["total_updates"]
        print(f"✅ Loaded (ε={self.epsilon:.3f}, updates={self.total_updates})")
        return True

    # ── State encoding ─────────────────────────────────────────────────────────

    def encode_state(self, obs):
        """Full board with 5 channels."""
        max_army = jnp.max(obs.armies)
        max_army = jnp.where(max_army > 0, max_army, 1.0)
        armies_norm = obs.armies / max_army

        return jnp.stack([
            obs.owned_cells.astype(jnp.float32),
            obs.opponent_cells.astype(jnp.float32),
            armies_norm,
            obs.mountains.astype(jnp.float32),
            obs.generals.astype(jnp.float32),
        ], axis=0)

    # ── Action selection ───────────────────────────────────────────────────────

    def act(self, obs, key):
        """
        Pick action from strongest owned cell.
        Network outputs Q-values for: [pass, up, down, left, right]
        """
        state = self.encode_state(obs)

        # Find strongest cell
        armies = obs.armies * obs.owned_cells
        if jnp.sum(armies) == 0:
            self.last_state = state
            self.last_action = 0
            return jnp.array([1, 0, 0, 0, 0])  # Pass

        # Add some randomness to cell selection (30% of time pick random strong cell)
        if random.random() < 0.3:
            flat_armies = armies.flatten()
            valid_indices = jnp.where(flat_armies > 1)[0]
            if len(valid_indices) > 0:
                chosen_flat = valid_indices[random.randint(0, len(valid_indices) - 1)]
                i, j = jnp.unravel_index(chosen_flat, armies.shape)
            else:
                i, j = jnp.unravel_index(jnp.argmax(armies), armies.shape)
        else:
            i, j = jnp.unravel_index(jnp.argmax(armies), armies.shape)

        # Get valid moves from this cell
        move_mask = compute_valid_move_mask(
            obs.armies, obs.owned_cells, obs.mountains
        )[i, j]

        # valid_actions: [True, up, down, left, right]
        valid_actions = [True] + list(move_mask)

        # Get Q-values from network
        q_values = self.network.apply(self.params, state)[0]  # (5,)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice([a for a in range(NUM_ACTIONS) if valid_actions[a]])
        else:
            # Mask invalid actions
            masked_q = jnp.where(
                jnp.array(valid_actions),
                q_values,
                -jnp.inf
            )
            action = int(jnp.argmax(masked_q))

        self.last_state = state
        self.last_action = action
        self.step_counter += 1

        # Convert to environment format
        if action == 0:
            return jnp.array([1, 0, 0, 0, 0])  # Pass
        else:
            return jnp.array([0, i, j, action - 1, 0])  # Move from (i,j) in direction

    def act_greedy(self, obs, key):
        """No exploration - for evaluation."""
        old_epsilon = self.epsilon
        self.epsilon = 0.0
        action = self.act(obs, key)
        self.epsilon = old_epsilon
        return action

    # ── Training ───────────────────────────────────────────────────────────────

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=(3,))
    def train_step(params, target_params, opt_state, optimizer,
                   states, actions, rewards, next_states, dones, gamma):
        def loss_fn(params):
            q_values = SpatialQNetwork(num_actions=NUM_ACTIONS).apply(params, states)
            batch_indices = jnp.arange(q_values.shape[0])
            predicted_q = q_values[batch_indices, actions]

            next_q = SpatialQNetwork(num_actions=NUM_ACTIONS).apply(target_params, next_states)
            max_next_q = jnp.max(next_q, axis=1)

            targets = rewards + gamma * max_next_q * (1 - dones)

            # Huber loss
            delta = predicted_q - targets
            huber_loss = jnp.where(
                jnp.abs(delta) <= 1.0,
                0.5 * delta ** 2,
                jnp.abs(delta) - 0.5
            )
            return jnp.mean(huber_loss)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        self.params, self.opt_state, loss = self.train_step(
            self.params, self.target_params, self.opt_state, self.optimizer,
            states, actions, rewards, next_states, dones, self.gamma
        )

        self.total_updates += 1
        if self.total_updates % self.target_update_freq == 0:
            self.target_params = self.params

        return float(loss)

    def update(self, prev_state, curr_state, reward, done=False):
        if self.last_action is not None:
            # Clip rewards
            reward = jnp.clip(reward, -10.0, 10.0)
            self.add_experience(prev_state, self.last_action, reward, curr_state, done)

        loss = 0.0
        if self.step_counter % self.train_freq == 0:
            loss = self.train()
        return loss

    # ── Reward ─────────────────────────────────────────────────────────────────

    def get_reward_from_states(self, prev_state, curr_state, player_id):
        prev_obs = get_observation(prev_state, player_id)
        curr_obs = get_observation(curr_state, player_id)

        reward = -0.01  # Small time penalty

        # Territory
        territory_gain = curr_obs.owned_land_count - prev_obs.owned_land_count
        reward += float(territory_gain) * 2.0

        # Army
        army_gain = curr_obs.owned_army_count - prev_obs.owned_army_count
        reward += float(army_gain) * 0.1

        # General captured (bad!)
        if prev_obs.generals.any() and not curr_obs.generals.any():
            reward -= 50.0

        # Victory!
        if curr_obs.opponent_land_count == 0 and prev_obs.opponent_land_count > 0:
            reward += 100.0

        # Enemy territory taken
        enemy_land_taken = prev_obs.opponent_land_count - curr_obs.opponent_land_count
        if enemy_land_taken > 0:
            reward += float(enemy_land_taken) * 3.0

        return reward

    # ── Episode end ────────────────────────────────────────────────────────────

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)