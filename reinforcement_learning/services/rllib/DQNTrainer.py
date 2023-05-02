from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG, GenericOffPolicyTrainer
from reinforcement_learning.services.rllib.DQNTFpolicy import DQNTFPolicy


# Build a DQN trainer, which uses the framework specific Policy
# determined in `get_policy_class()` above.
DQNTrainer = GenericOffPolicyTrainer.with_updates(
    name="DQN", default_policy=DQNTFPolicy, default_config=DEFAULT_CONFIG)
