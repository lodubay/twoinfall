r"""
This script runs a multi-zone model with the parameters specified by command-
line arguments.

Run ``python -m multizone.py --help`` for more info.
"""

import argparse
from . import _globals
from . import src

_MIGRATION_MODELS_ = ["diffusion", "linear", "post-process", "sudden", 
                      "gaussian", "none"]
_EVOLUTION_MODELS_ = ["static", "insideout", "lateburst", "outerburst",
                      "twoinfall", "earlyburst"]
_DELAY_MODELS_ = ["powerlaw", "plateau", "prompt", "exponential", "triple",
                  "greggio05_single", "greggio05_double"]
_YIELD_SETS_ = ["F04", "JW20", "J21", "C22", "W23"]

def parse():
    r"""
    Parse the command line arguments using argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description = "The parameters of the Milky Way models to run.")

    parser.add_argument("-f", "--force",
        help = "Force overwrite existing VICE outputs of the same name.",
        action = "store_true")

    parser.add_argument("--migration",
        help = "The migration model to assume. (Default: gaussian)",
        type = str,
        choices = _MIGRATION_MODELS_,
        default = "gaussian")

    parser.add_argument("--evolution",
        help = "The evolutionary history to assume (Default: twoinfall)",
        type = str,
        choices = _EVOLUTION_MODELS_,
        default = "twoinfall")

    parser.add_argument("--RIa",
        help = "The SN Ia delay-time distribution to assume (Default: plateau)",
        type = str,
        choices = _DELAY_MODELS_,
        default = "plateau")

    parser.add_argument("--RIa-params",
        help = "Parameters for the SN Ia delay-time distribution separated by \
underscores. (Default: '')",
        type = str,
        default = "")

    parser.add_argument("--minimum-delay",
         help = "The minimum SN Ia delay time in Gyr (Default: 0.04)",
         type = float,
         default = _globals.MIN_RIA_DELAY)

    parser.add_argument("--dt",
        help = "Timestep size in Gyr. (Default: 0.01)",
        type = float,
        default = _globals.DT)

    parser.add_argument("--nstars",
        help = """Number of stellar populations per zone per timestep. \
(Default: 8)""",
        type = int,
        default = _globals.NSTARS)

    parser.add_argument("--name",
        help = "The name of the output simulations (Default: 'diskmodel')",
        type = str,
        default = '../data/multizone/diskmodel')

    parser.add_argument("--elements",
        help = """Elements to simulation the enrichment for separated by \
underscores. (Default: \"fe_o\")""",
        type = str,
        default = "_".join(_globals.ELEMENTS))

    parser.add_argument("--zonewidth",
        help = "The width of each annulus in kpc. (Default: 0.1)",
        type = float,
        default = _globals.ZONE_WIDTH)
    
    parser.add_argument("--yields",
        help = "The nucleosynthetic yield set to use. (Default: 'J21')",
        type = str,
        choices = _YIELD_SETS_,
        default = "J21")
    
    parser.add_argument("--seed", 
                        help = "Seed for the random number generator.",
                        type = int,
                        default = _globals.RANDOM_SEED)
    
    parser.add_argument("--gasvelocity",
                        help = "Radial gas velocity in km/s.",
                        type = float,
                        default = 0.)
    
    parser.add_argument("--no-outflows",
                        help = "Disable mass-loaded outflows.",
                        action = "store_true")

    return parser


def model(args):
    r"""
    Get the milkyway object corresponding to the desired simulation.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed via argparse.
    """
    # Parse RIa params into dict
    RIa_kwargs = {}
    if '=' in args.RIa_params:
        for p in args.RIa_params.split('_'):
            key, value = p.split('=')
            RIa_kwargs[key] = float(value)
    config = src.config(
        timestep_size = args.dt,
        star_particle_density = args.nstars,
        zone_width = args.zonewidth,
        elements = args.elements.split('_')
    )
    kwargs = dict(
        name = args.name,
        spec = args.evolution,
        RIa = args.RIa,
        RIa_kwargs = RIa_kwargs,
        delay = args.minimum_delay,
        yields = args.yields,
        seed = args.seed,
        radial_gas_velocity = args.gasvelocity,
        outflows = not args.no_outflows
    )
    if args.migration == "post-process":
        kwargs["simple"] = True
    else:
        kwargs["migration_mode"] = args.migration
    return src.diskmodel.from_config(config, **kwargs)


def main():
    r"""
    Runs the script.
    """
    parser = parse()
    args = parser.parse_args()
    model_ = model(args)
    model_.run([_ * model_.dt for _ in range(round(
        _globals.END_TIME / model_.dt) + 1)],
        overwrite = args.force, pickle = True)


if __name__ == "__main__": 
    main()
