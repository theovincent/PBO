import sys
import argparse
from json import load, dump


def edit_json_cli(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Edit a parameter in a json file.")
    parser.add_argument(
        "-f",
        "--file",
        help="Path to the file.",
        required=True,
    )
    parser.add_argument("-k", "--key", help="Key to edit.", required=True)
    parser.add_argument("-v", "--value", help="New value.", required=True, type=float)

    args = parser.parse_args(argvs)

    with open(args.file, "r") as parameters_file:
        parameters = load(parameters_file)

    if type(parameters[args.key]) == float:
        parameters[args.key] = float(args.value)
    else:
        parameters[args.key] = int(args.value)

    with open(args.file, "w") as parameters_file:
        dump(parameters, parameters_file)
