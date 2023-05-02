from gym.envs.registration import register
register(
    id="tankgym-v0",
    entry_point="reinforcement_learning.tank_plc_gym.envs.tank_gym:TankGym",
)
