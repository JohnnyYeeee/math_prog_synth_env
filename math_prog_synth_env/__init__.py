from gym.envs.registration import register

register(
    id="math-env-v0",
    entry_point="math_prog_synth_env.envs:MathEnv",
)
