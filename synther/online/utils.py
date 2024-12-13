import gym
import numpy as np
from gym.wrappers.flatten_observation import FlattenObservation
from redq.algos.core import ReplayBuffer
from synther.online.redq_rlpd_agent import REDQRLPDAgent


def wrap_gym(env: gym.Env, rescale_actions: bool = True) -> gym.Env:
    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)

    return env


# Make transition dataset from REDQ replay buffer.
def make_inputs_from_replay_buffer(
        replay_buffer: ReplayBuffer,
        model_terminals: bool = False,
) -> np.ndarray:
    ptr_location = replay_buffer.ptr
    obs = replay_buffer.obs1_buf[:ptr_location]
    actions = replay_buffer.acts_buf[:ptr_location]
    next_obs = replay_buffer.obs2_buf[:ptr_location]
    rewards = replay_buffer.rews_buf[:ptr_location]
    inputs = [obs, actions, rewards[:, None], next_obs]
    if model_terminals:
        terminals = replay_buffer.done_buf[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None])
    return np.concatenate(inputs, axis=1)

def is_action_ok(env: gym.Env, act: np.ndarray):
    act_low = env.action_space.low
    act_high = env.action_space.high
    for value, lower, upper in zip(act, act_low, act_high):  
        if value < lower or value > upper:  
            return False 
    return True


def action_generator(agent: REDQRLPDAgent, env: gym.Env, obs: np.ndarray):
    act = agent.get_test_action(obs)
    act_low = env.action_space.low
    act_high = env.action_space.high
    ep = 0.05
    rg = ep * (act_high - act_low)
    act_len = len(act_low)
    random_a = np.random.uniform(-1, 1, size=act_len)
    act_1 = act + np.dot(rg, random_a)
    if is_action_ok(env, act_1):
        act = act_1

    return act

