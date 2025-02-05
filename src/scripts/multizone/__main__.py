r"""
This script runs a multi-zone model with the parameters specified by command-
line arguments.

Run ``python -m multizone.py --help`` for more info.
"""

import argparse

from . import _globals
from . import src
import paths

_MIGRATION_MODELS_ = [
    "diffusion", 
    "linear", 
    "post-process", 
    "sudden", 
    "gaussian", 
    "none"
]
_EVOLUTION_MODELS_ = [
    "static", 
    "insideout", 
    "lateburst", 
    "outerburst",
    "earlyburst", 
    "oneinfall",
    "static_infall",
    "twoinfall", 
    "twoinfall_linvar", 
    "twoinfall_expvar", 
    "twoinfall_inner"
]
_DELAY_MODELS_ = [
    "powerlaw", 
    "plateau", 
    "prompt", 
    "exponential", 
    "triple",
    "greggio05_single", 
    "greggio05_double"
]
_YIELD_SETS_ = [
    "F04", "JW20", "J21", "C22", "W24", "W24mod", "yZ1", "yZ2", "yZ3"
]
_OUTFLOW_PRESCRIPTIONS_ = [
    "J21", # aka vice.milkyway.default_mass_loading
    "equilibrium",
    "bespoke", # different prescriptions tuned to each yield set
    "none"
]

def parse():
    r"""
    Parse the command line arguments using argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description = "The parameters of the Milky Way models to run."
    )
    parser.add_argument("-f", "--force",
        help = "Force overwrite existing VICE outputs of the same name.",
        action = "store_true"
    )
    parser.add_argument("-p", "--pickle",
        help = "Save functional attributes along with VICE output.",
        action = "store_true"
    )
    parser.add_argument("--migration",
        help = "The migration model to assume. (Default: gaussian)",
        type = str,
        choices = _MIGRATION_MODELS_,
        default = "gaussian"
    )
    parser.add_argument("--migration-time-dependence",
        help = "Power on the time-dependence of radial migration speed \
(Gaussian migration only; default: 0.33)",
        type = float,
        default = 0.33)
    parser.add_argument("--migration-radial-dependence",
        help = "Power on the radial dependence of radial migration speed \
(Gaussian migration only; default: 0.61)",
        type = float,
        default = 0.61)
    parser.add_argument("--migration-strength",
        help = "Coefficient for the strength of radial migration in kpc \
(Gaussian migration only; default: 2.68)",
        type = float,
        default = 2.68)
    parser.add_argument("--evolution",
        help = "The evolutionary history to assume (Default: twoinfall)",
        type = str,
        choices = _EVOLUTION_MODELS_,
        default = "twoinfall"
    )
    parser.add_argument("--RIa",
        help = "The SN Ia delay-time distribution to assume (Default: plateau)",
        type = str,
        choices = _DELAY_MODELS_,
        default = "plateau"
    )
    parser.add_argument("--RIa-params",
        help = "Parameters for the SN Ia delay-time distribution separated by \
underscores. (Default: '')",
        type = str,
        default = ""
    )
    parser.add_argument("--minimum-delay",
        help = "The minimum SN Ia delay time in Gyr (Default: 0.04)",
        type = float,
        default = _globals.MIN_RIA_DELAY
    )
    parser.add_argument("--dt",
        help = "Timestep size in Gyr. (Default: 0.01)",
        type = float,
        default = _globals.DT
    )
    parser.add_argument("--nstars",
        help = """Number of stellar populations per zone per timestep. \
(Default: 8)""",
        type = int,
        default = _globals.NSTARS
    )
    parser.add_argument("--name",
        help = "The name of the output simulations (Default: 'diskmodel')",
        type = str,
        default = "diskmodel"
    )
    parser.add_argument("--elements",
        help = """Elements to simulation the enrichment for separated by \
underscores. (Default: \"fe_o\")""",
        type = str,
        default = "_".join(_globals.ELEMENTS)
    )
    parser.add_argument("--zonewidth",
        help = "The width of each annulus in kpc. (Default: 0.1)",
        type = float,
        default = _globals.ZONE_WIDTH
    )
    parser.add_argument("--yields",
        help = "The nucleosynthetic yield set to use. (Default: 'yZ1')",
        type = str,
        choices = _YIELD_SETS_,
        default = "yZ1"
    )
    parser.add_argument("--seed", 
        help = "Seed for the random number generator.",
        type = int,
        default = _globals.RANDOM_SEED
    )
    parser.add_argument("--radial-gas-velocity",
        help = "Radial gas velocity in km/s, negative for \
an inward flow (default: 0).",
        type = float,
        default = 0.
    )
    parser.add_argument("--outflows",
        help = "The outflow prescription as a function of Galactocentric radius.",
        type = str,
        choices = _OUTFLOW_PRESCRIPTIONS_,
        default = "equilibrium"
    )
    parser.add_argument("--pre-enrichment",
        help = "The [X/H] abundance of the infalling gas at late times. \
If -inf, infalling gas is always pristine. (Default: -inf).",
        default = float("-inf"),
        type = float
    )
    parser.add_argument("--pre-alpha-enhancement",
        help = "The [alpha/M] enhancement of the infalling gas at late times. \
(Default: 0.0).",
        default = 0.0,
        type = float
    )

    return parser


def model(args):
    r"""
    Get the milkyway object corresponding to the desired simulation.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments parsed via argparse.
    """
    # Create output dir (and parents) if it doesn't exist
    fullpath = paths.multizone / args.name
    if not fullpath.parents[0].exists():
        fullpath.parents[0].mkdir(parents=True)
    # Save command-line arguments to a file
    with open(str(fullpath) + "_args.txt", "w") as f:
        f.write(args.name + "\n")
        f.writelines(["%s: %s\n" % (k, v) for k, v in vars(args).items() if k != "name"])
        f.write("thick_to_thin_ratio: %s\n" % _globals.THICK_TO_THIN_RATIO)
    # Parse RIa params into dict
    RIa_kwargs = {}
    if '=' in args.RIa_params:
        for p in args.RIa_params.split("_"):
            key, value = p.split("=")
            RIa_kwargs[key] = float(value)
    config = src.config(
        timestep_size = args.dt,
        star_particle_density = args.nstars,
        zone_width = args.zonewidth,
        elements = args.elements.split("_")
    )
    kwargs = dict(
        name = str(fullpath),
        spec = args.evolution,
        RIa = args.RIa,
        RIa_kwargs = RIa_kwargs,
        delay = args.minimum_delay,
        yields = args.yields,
        seed = args.seed,
        radial_gas_velocity = args.radial_gas_velocity,
        outflow_spec = args.outflows,
        pre_enrichment = args.pre_enrichment,
        pre_alpha_enhancement = args.pre_alpha_enhancement,
        migration_time_dep = args.migration_time_dependence,
        migration_radius_dep = args.migration_radial_dependence,
        migration_strength = args.migration_strength,
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
        overwrite = args.force, pickle = args.pickle)


if __name__ == "__main__": 
    main()
