# characterize_chemicals/cli.py

import os
import argparse
import asyncio
import logging
from typing import List

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from .chem_agent import MoleculePropertyAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-planned RDKit + xTB molecular characterization agent."
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Planner LLM model name (Hugging Face ID).",
    )

    parser.add_argument(
        "--smiles",
        "-s",
        nargs="+",
        default=["CCO", "c1ccccc1", "CC(=O)O"],
        help="One or more SMILES strings to characterize.",
    )

    parser.add_argument(
        "--props",
        "-p",
        nargs="+",
        default=["logP", "dipole_moment", "solvation_free_energy"],
        help="Target properties (hint to the planner).",
    )

    parser.add_argument(
        "--accuracy-profile",
        "-a",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "high"],
        help="Accuracy/speed profile hint for the planner.",
    )

    parser.add_argument(
        "--max-wallclock-minutes",
        type=float,
        default=10.0,
        help="Maximum wallclock minutes per molecule.",
    )

    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Expose model name to the agent via env var (simplest from Academy side)
    os.environ["PLANNER_MODEL"] = args.model

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        chem_handle = await manager.launch(MoleculePropertyAgent)

        smiles_list: List[str] = args.smiles
        target_properties: List[str] = args.props

        logging.info(
            "Running MoleculePropertyAgent with model=%s on SMILES=%s props=%s",
            args.model,
            smiles_list,
            target_properties,
        )

        for smiles in smiles_list:
            res = await chem_handle.compute_properties(
                molecule_smiles=smiles,
                target_properties=target_properties,
                accuracy_profile=args.accuracy_profile,
                max_wallclock_minutes=args.max_wallclock_minutes,
            )
            print("=" * 60)
            print(f"SMILES: {smiles}")
            print("Status:", res["status"])
            print("Properties:", res["properties"])
            print()

        await manager.shutdown(chem_handle, blocking=True)

    return 0


def main() -> None:
    args = parse_args()
    asyncio.run(_async_main(args))
