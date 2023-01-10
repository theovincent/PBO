import argparse


def addparse(
    parser: argparse.ArgumentParser,
    seed: bool = False,
    architecture: bool = False,
    validation_bellman_iterations: bool = False,
) -> None:
    parser.add_argument(
        "-e",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-b",
        "--max_bellman_iterations",
        help="Maximum number of Bellman iteration.",
        type=int,
        required=True,
    )

    if seed:
        parser.add_argument(
            "-s",
            "--seed",
            help="Seed of the training.",
            type=int,
            required=True,
        )

    if architecture:
        parser.add_argument(
            "-a",
            "--architecture",
            help="Class of the PBO.",
            choices=["linear", "max_linear", "custom_linear", "deep"],
            required=True,
        )
        parser.add_argument(
            "-c", "--conv", help="PBO made out of convolutional layers or not.", default=False, action="store_true"
        )

    if validation_bellman_iterations:
        parser.add_argument(
            "-v",
            "--validation_bellman_iterations",
            help="Number of Bellman iteration to validate on.",
            default=10,
            type=int,
        )
