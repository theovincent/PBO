def print_info(experiment_name: str, algorithm: str, environment_name: str, bellman_iteration_scope: int, seed: int):
    print(f"-------- {experiment_name} --------")
    print(
        f"Training {algorithm} on {environment_name} with {bellman_iteration_scope} Bellman iterations at a time and seed {seed}..."
    )
