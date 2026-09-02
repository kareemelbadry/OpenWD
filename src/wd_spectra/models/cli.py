"""Command-line interface shared by the four one-shot examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import ModelData, ModelResult, load_atmosphere_checkpoint, save_model_result
from .stellar import (
    DAConfig,
    DABConfig,
    DBConfig,
    DZConfig,
    compute_da,
    compute_dab,
    compute_db,
    compute_dz,
)


def _assignment(value: str) -> tuple[str, float]:
    try:
        element, raw = value.split("=", 1)
        return element.strip(), float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected Element=value") from exc


def _quicklook(result: ModelResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = result.spectrum.wavelength_angstrom
    flux = result.spectrum.surface_flux_lambda
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].loglog(wavelength, wavelength * flux, color="#d55e00", lw=0.9)
    axes[0].set_ylabel(r"$\lambda F_\lambda$")
    axes[0].set_title(
        f"{result.spectral_type}: "
        f"Teff={result.atmosphere.effective_temperature:.0f} K, "
        f"log g={result.atmosphere.logg:.2f}"
    )
    optical = (wavelength >= 3400.0) & (wavelength <= 7500.0)
    if np.any(optical):
        normalizer = np.nanpercentile(flux[optical], 95.0)
        axes[1].plot(wavelength[optical], flux[optical] / normalizer,
                     color="#d55e00", lw=0.7)
        axes[1].set_xlim(3400.0, 7500.0)
        axes[1].set_ylabel("Relative surface flux")
    axes[1].set_xlabel(r"Vacuum wavelength [$\AA$]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def one_shot_main(spectral_type: str) -> None:
    """Run a DA, DB, DAB, or DZ atmosphere and formal spectrum."""

    kind = spectral_type.upper()
    if kind not in {"DA", "DB", "DAB", "DZ"}:
        raise ValueError(f"unsupported spectral type {spectral_type!r}")
    parser = argparse.ArgumentParser(
        description=f"Produce one self-consistent {kind} atmosphere and spectrum."
    )
    parser.add_argument("--teff", type=float)
    parser.add_argument("--logg", type=float)
    parser.add_argument("--quality", choices=("quick", "standard", "production"),
                        default="standard")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="optional external data root; bundled data are the default")
    parser.add_argument("--output", type=Path,
                        default=Path("results/one-shot") / kind.lower())
    parser.add_argument("--wavelength-min", type=float)
    parser.add_argument("--wavelength-max", type=float)
    parser.add_argument("--wavelength-step", type=float)
    parser.add_argument("--restart-atmosphere", type=Path)
    if kind in {"DA", "DAB"}:
        parser.add_argument("--lyman-profiles", choices=("allard", "stark"),
                            default="allard")
    if kind == "DA":
        parser.add_argument("--h3plus-partition",
                            choices=("neale-tennyson-1995", "none"),
                            default="neale-tennyson-1995")
    if kind in {"DAB", "DZ"}:
        parser.add_argument("--log-h-he", type=float,
                            default=-2.0 if kind == "DAB" else -6.16)
    if kind == "DZ":
        parser.add_argument("--abundance", action="append", type=_assignment,
                            help="replace defaults with Element=log10(N/He)")
        parser.add_argument("--dense-helium-eos", choices=("ideal", "reos3"),
                            default=DZConfig().dense_helium_eos)
        parser.add_argument("--strong-line-atomic-data",
                            choices=("stout", "nist-asd"),
                            default=DZConfig().strong_line_atomic_data)
    args = parser.parse_args()

    supplied_grid = (args.wavelength_min, args.wavelength_max, args.wavelength_step)
    if any(value is not None for value in supplied_grid):
        if not all(value is not None for value in supplied_grid):
            parser.error("supply all three wavelength options together")
        if (args.wavelength_min <= 0 or args.wavelength_max <= args.wavelength_min
                or args.wavelength_step <= 0):
            parser.error("wavelength bounds and step must be positive and increasing")
        wavelength = np.arange(args.wavelength_min,
                               args.wavelength_max + 0.5 * args.wavelength_step,
                               args.wavelength_step)
    else:
        wavelength = None

    data = ModelData.default(args.data_root)
    if kind == "DA":
        defaults = DAConfig()
        config = DAConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            quality=args.quality,
            lyman_profile_source=args.lyman_profiles,
            h3plus_partition_model=args.h3plus_partition,
        )
        composition = "hydrogen"
        compute = compute_da
    elif kind == "DB":
        defaults = DBConfig()
        config = DBConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            quality=args.quality,
        )
        composition = "helium"
        compute = compute_db
    elif kind == "DAB":
        defaults = DABConfig()
        config = DABConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            log_hydrogen_to_helium=args.log_h_he,
            quality=args.quality,
            lyman_profile_source=args.lyman_profiles,
        )
        composition = "mixed"
        compute = compute_dab
    else:
        defaults = DZConfig()
        config = DZConfig(
            effective_temperature=defaults.effective_temperature if args.teff is None else args.teff,
            logg=defaults.logg if args.logg is None else args.logg,
            abundances=dict(args.abundance) if args.abundance else defaults.abundances,
            log_hydrogen_abundance=args.log_h_he,
            quality=args.quality,
            dense_helium_eos=args.dense_helium_eos,
            strong_line_atomic_data=args.strong_line_atomic_data,
        )
        composition = "helium"
        compute = compute_dz

    atmosphere = None
    if args.restart_atmosphere is not None:
        checkpoint_kwargs = {}
        if kind == "DAB":
            checkpoint_kwargs["log_hydrogen_to_helium"] = config.log_hydrogen_to_helium
        if kind == "DA":
            molecular = config.effective_temperature <= 12_000.0
            checkpoint_kwargs.update(
                include_molecules=molecular,
                include_negative_hydrogen=molecular,
                trihydrogen_ion_partition_model=(
                    None if config.h3plus_partition_model == "none"
                    else config.h3plus_partition_model
                ),
            )
        atmosphere = load_atmosphere_checkpoint(
            args.restart_atmosphere,
            config.effective_temperature,
            config.logg,
            composition,
            **checkpoint_kwargs,
        )
    result = compute(
        config,
        wavelength,
        data=data,
        initial_atmosphere=atmosphere,
        relax_atmosphere=atmosphere is None,
    )
    directory = save_model_result(result, args.output)
    _quicklook(result, directory / "spectrum.png")
    print(f"Wrote {directory}")
