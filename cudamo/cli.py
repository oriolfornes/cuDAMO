"""Command line interface."""

import sys

import click

from cudamo.damo import DEFAULT_WIDTHS, damo, pwm_to_pfm
from cudamo.io import (open_file, read_motif, write_jaspar, write_meme,
                       write_pwm)

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
@click.option(
    "-w", "--optimize-width",
    help="Also optimize the motif width: try several widths (see --widths) "
         "and keep the one that does best on held-out sequences.",
    is_flag=True,
)
@click.option(
    "--widths",
    help="Comma-separated widths to try; implies -w. "
         f"[default: {','.join(map(str, DEFAULT_WIDTHS))}]",
    callback=lambda ctx, param, value: None if value is None else [
        int(k) for k in value.split(",")],
)
@click.option(
    "-m", "--motif",
    help="Motif to use when the MEME file has several (name, or its ID).",
)
@click.option(
    "--oformat",
    help="Output format: the log PWM (one line per position, A C G T), or "
         "the motif as frequencies in JASPAR or MEME format.",
    type=click.Choice(["pwm", "jaspar", "meme"]),
    default="pwm",
    show_default=True,
)
@click.option(
    "--scale",
    help="With --oformat jaspar or meme: factor applied to the PWM before "
         "converting it to frequencies; above 1 sharpens the columns, "
         "below 1 flattens them. [default: 1]",
    type=click.FloatRange(min=0, min_open=True),
)
def main(**args):
    """Optimize a JASPAR or MEME motif to separate positive from negative sequences.

    The input format is detected from the file. The output is the log PWM
    by default, with one line per position and four columns (A, C, G, T).
    With --oformat jaspar or meme it is a motif of frequencies instead: the
    PWM weights are read as energies (kT) and converted with a Boltzmann
    distribution, as in DiMO.
    """
    if args["scale"] is not None and args["oformat"] == "pwm":
        raise click.UsageError(
            "--scale only applies with --oformat jaspar or meme")

    widths = args["widths"]
    if widths is None and args["optimize_width"]:
        widths = list(DEFAULT_WIDTHS)

    pwm = damo(args["positive_seqs"], args["negative_seqs"],
        args["jaspar_profile"], args["debugging"],
        iterations=args["iterations"], motif_name=args["motif"],
        widths=widths)

    if args["oformat"] == "pwm":
        def write(handle):
            write_pwm(pwm, handle)
    else:
        name, _ = read_motif(args["jaspar_profile"], args["motif"])
        pfm = pwm_to_pfm(pwm, args["scale"] or 1.0)
        writer = write_jaspar if args["oformat"] == "jaspar" else write_meme

        def write(handle):
            writer(pfm, handle, name)

    if args["output_file"] is None:
        write(sys.stdout)
    else:
        with open_file(args["output_file"], "w") as handle:
            write(handle)
