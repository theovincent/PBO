def print_info(
    experiment_name: str,
    algorithm: str,
    environment_name: str,
    max_bellman_iterations: int,
    seed: int = None,
    train: bool = True,
):
    print(f"\n\n-------- {experiment_name} --------")
    if train:
        print(f"Training {algorithm} on {environment_name} with {max_bellman_iterations} Bellman iterations", end="")
    else:
        print(f"Evaluating {algorithm} on {environment_name} with {max_bellman_iterations} Bellman iterations", end="")

    if seed is not None:
        print(f" and seed {seed}...\n")
    else:
        print("...")
