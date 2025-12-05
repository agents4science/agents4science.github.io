import os
import asyncio
import logging
import argparse
from typing import List

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from chem_agent import MoleculePropertyAgent

from academy.exchange.cloud.client import HttpExchangeFactory
from globus_compute_sdk import Executor as GCExecutor

EXCHANGE_ADDRESS = "https://exchange.academy-agents.org"


# Optional: tame OpenMP issues on macOS / conda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoleculePropertyAgent with optional model and SMILES arguments.",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help=(
            "HF model id for the local planner LLM "
            "(default: microsoft/Phi-3.5-mini-instruct)"
        ),
    )

    parser.add_argument(
        "--smiles",
        "-s",
        nargs="+",
        default=["CCO", "c1ccccc1", "CC(=O)O"],
        help=(
            "One or more SMILES strings to evaluate. "
            "Example: -s CCO c1ccccc1 CC(=O)O"
        ),
    )

    parser.add_argument(
        "--props",
        "-p",
        nargs="+",
        default=["logP", "dipole_moment", "solvation_free_energy"],
        help=(
            "Target properties to compute for each molecule. "
            "Default: logP dipole_moment solvation_free_energy"
        ),
    )

    parser.add_argument(
        "--accuracy-profile",
        "-a",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "high"],
        help="Accuracy/speed tradeoff hint for the planner (default: balanced).",
    )

    return parser.parse_args()


async def main(args) -> int:
    logging.basicConfig(level=logging.INFO)

    # 1. Choose an executor ("launcher") based on environment
    if "CHEM_ENDPOINT_ID" in os.environ:
        # Launch agents on a Globus Compute endpoint
        executor = GCExecutor(os.environ["CHEM_ENDPOINT_ID"])
    else:
        # Fallback: run agents in a local process pool
        mp_context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=6,
            initializer=logging.basicConfig,
            mp_context=mp_context,
        )

    # 2. Use HttpExchangeFactory + executors=... (like run-04.py)
    async with await Manager.from_exchange_factory(
        factory=HttpExchangeFactory(
            EXCHANGE_ADDRESS,
            auth_method="globus",
        ),
        executors=executor,
    ) as manager:
        # Launch your chem agent as usual
        chem_handle = await manager.launch(MoleculePropertyAgent)

        # Call actions on the agent
        for smiles in args.smiles:
            res = await chem_handle.compute_properties(
                molecule_smiles=smiles,
                target_properties=args.props,
                accuracy_profile=args.accuracy_profile,
                max_wallclock_minutes=10,
            )
            print("SMILES:", smiles)
            print("Status:", res["status"])
            print("Properties:", res["properties"])
            print()


async def main_OLD(args: argparse.Namespace) -> None:
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
    ) as manager:
        # Launch the agent with the chosen model
        chem_handle = await manager.launch(MoleculePropertyAgent, args=(args.model,))
        smiles_list: List[str] = args.smiles
        target_properties: List[str] = args.props

        logging.info(
            "Running MoleculePropertyAgent with model=%s on SMILES=%s props=%s",
            args.model,
            smiles_list,
            target_properties,
        )

        results = []
        # Sequential calls (keeps planner LLM happy on a single M1)
        for smiles in smiles_list:
            res = await chem_handle.compute_properties(
                molecule_smiles=smiles,
                target_properties=target_properties,
                accuracy_profile=args.accuracy_profile,
                max_wallclock_minutes=10,
            )
            results.append(res)

        for smiles, res in zip(smiles_list, results, strict=True):
            print("=" * 60)
            print(f"SMILES: {smiles}")
            print("Status:", res["status"])
            print("Properties:", res["properties"])
            print("Plan steps:", len(res["plan_used"]["steps"]))
            print()

        await manager.shutdown(chem_handle, blocking=True)


if __name__ == "__main__":
    cli_args = parse_args()
    # Make model name visible to the agent via env var
    os.environ["PLANNER_MODEL"] = cli_args.model
    asyncio.run(main(cli_args))

