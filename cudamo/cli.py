"""Command line interface."""

import sys

import click

from cudamo.damo import damo
from cudamo.io import open_file, write_pwm

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "positive_seqs",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "negative_seqs",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "jaspar_profile",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-d", "--debugging",
    help="Use only the first 1,000 sequences of each file.",
    is_flag=True,
)
@click.option(
    "-i", "--iterations",
    help="Number of optimization iterations.",
    type=int,
    default=500,
    show_default=True,
)
@click.option(
    "-o", "--output-file",
    help="Output file (.gz to compress). [default: stdout]",
    type=click.Path(resolve_path=True),
)
def main(**args):
    """Optimize a JASPAR motif to separate positive from negative sequences.

    The log PWM is written with one line per position and four columns
    (A, C, G, T).
    """
    pwm = damo(args["positive_seqs"], args["negative_seqs"],
        args["jaspar_profile"], args["debugging"],
        iterations=args["iterations"])

    if args["output_file"] is None:
        write_pwm(pwm, sys.stdout)
    else:
        with open_file(args["output_file"], "w") as handle:
            write_pwm(pwm, handle)

