"""Command line interface."""

import sys

import click

from daimo.damo import (DEFAULT_SCALE, _parse_width_ranges,
                        damo, pwm_to_pfm)
from daimo.io import (open_file, read_motif, write_jaspar, write_meme,
                       write_pwm)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


def _check_widths(value):
    """Validate the syntax of --widths; open-ended ranges are expanded later."""
    if value is not None:
        try:
            _parse_width_ranges(value)
        except ValueError as e:
            raise click.BadParameter(str(e)) from None

    return value


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
    "seed_motif",
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
    "--widths",
    help="Widths to try, as a list in the style of cut -f: 8,10,12 or 5-19 or "
         "5-8,12,15- (-M is 1 to M, N- is N up to the sequence length). "
         "[default: the width of the seed motif]",
    callback=lambda ctx, param, value: _check_widths(value),
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
         "below 1 flattens them. 0.1 keeps the ranking of sites "
         "through the rounding of the output file; 1 is DiMO's unscaled "
         "conversion and gives nearly one-hot columns.",
    type=click.FloatRange(min=0, min_open=True),
    default=DEFAULT_SCALE,
    show_default=True,
)
def main(**args):
    """Optimize a JASPAR or MEME motif to separate positive from negative sequences.

    The input format is detected from the file. The output is the log PWM
    by default, with one line per position and four columns (A, C, G, T).
    With --oformat jaspar or meme it is a motif of frequencies instead: the
    PWM weights are read as energies (kT) and converted with a Boltzmann
    distribution, as in DiMO, after scaling them by 0.1 (see --scale).
    """
    ctx = click.get_current_context()
    scale_given = (ctx.get_parameter_source("scale")
                   != click.core.ParameterSource.DEFAULT)
    if scale_given and args["oformat"] == "pwm":
        raise click.UsageError(
            "--scale only applies with --oformat jaspar or meme")

    pwm = damo(args["positive_seqs"], args["negative_seqs"],
        args["seed_motif"], args["debugging"],
        iterations=args["iterations"], motif_name=args["motif"],
        widths=args["widths"])

    if args["oformat"] == "pwm":
        def write(handle):
            write_pwm(pwm, handle)
    else:
        name, _ = read_motif(args["seed_motif"], args["motif"])
        pfm = pwm_to_pfm(pwm, args["scale"])
        writer = write_jaspar if args["oformat"] == "jaspar" else write_meme

        def write(handle):
            writer(pfm, handle, name)

    if args["output_file"] is None:
        write(sys.stdout)
    else:
        with open_file(args["output_file"], "w") as handle:
            write(handle)
