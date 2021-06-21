from gym.envs.registration import register

register(
    id="math-env-v0",
    entry_point="dm_math_gym_env.envs:MathEnv",
)
