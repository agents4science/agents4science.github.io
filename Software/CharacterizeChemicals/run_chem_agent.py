import os
import asyncio
import logging
import argparse
from typing import List

from academy.manager import Manager

from characterize_chemicals.chem_agent import MoleculePropertyAgent

#EXCHANGE_ADDRESS = "https://exchange.academy-agents.org"

EXCHANGE_PORT = 8000 

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

    parser.add_argument(
        "--max-wallclock-minutes",
        "-w",
        type=float,
        default=10.0,
        help="Maximum wallclock minutes per molecule.",
    )

    return parser.parse_args()


async def launch_chem_agent(manager, args):
    """
    This function assumes it's inside an event loop and can use await.
    """

    chem_handle = await manager.launch(MoleculePropertyAgent)

    results = []
    # We want this to be parallel I assume?
    for smiles in args.smiles:
        res = await chem_handle.compute_properties(
            molecule_smiles=smiles,
            target_properties=args.props,
            accuracy_profile=args.accuracy_profile,
            max_wallclock_minutes=args.max_wallclock_minutes,
        )
        results.append((smiles, res))
    return results


async def main(args) -> int:
    logging.basicConfig(level=logging.INFO)

    if "EXCHANGE_PORT" in os.environ:
        # Use Parsl executor
        from parsl.concurrent import ParslPoolExecutor
        from parsl.configs.htex_local import config
        from academy.exchange.cloud import spawn_http_exchange

        from parsl.concurrent import ParslPoolExecutor
        from parsl.config import Config
        from parsl.executors.threads import ThreadPoolExecutor
        from academy.exchange.cloud.client import HttpExchangeFactory

        print('RUN_CHEM_AGENT: Using Parsl executor')
        with spawn_http_exchange("localhost", EXCHANGE_PORT) as factory:
            # e.g. Parsl executor
            executor = ParslPoolExecutor(
                config=Config(
                    executors=[ThreadPoolExecutor(max_threads=3)]
                )

            )
            async with await Manager.from_exchange_factory(
                factory=factory,
                executors=executor,
            ) as manager:
                results = await launch_chem_agent(manager, args)
                for smiles, res in results:
                    print("=" * 60)
                    print("SMILES:", smiles)
                    print("Status:", res["status"])
                    print("Properties:", res["properties"])
                    print()

    elif "EXCHANGE_ADDRESS" in os.environ:
        # Run agents in a local process pool
        from academy.exchange.cloud.client import HttpExchangeFactory

        print('RUN_CHEM_AGENT: Using local multiprocessing executor')
        factory=HttpExchangeFactory(
            EXCHANGE_ADDRESS,
            auth_method="globus",
        ),
        mp_context = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(
            max_workers=3,
            initializer=logging.basicConfig,
            mp_context=mp_context,
        )

    #elif "CHEM_ENDPOINT_ID" in os.environ:
        # Launch agents on a Globus Compute endpoint
        #from globus_compute_sdk import Executor as GCExecutor
        #executor = GCExecutor(os.environ["CHEM_ENDPOINT_ID"])

    else:
        # Launch the agent locally
        from academy.exchange import LocalExchangeFactory
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        print('RUN_CHEM_AGENT: Using local executor')
        async with await Manager.from_exchange_factory(
            factory=LocalExchangeFactory(),
        ) as manager:
            # Launch the agent with embedded planner
            results = await launch_chem_agent(manager, args)
            for smiles, res in results:
                print("=" * 60)
                print("SMILES:", smiles)
                print("Status:", res["status"])
                print("Properties:", res["properties"])
                print()


if __name__ == "__main__":
    cli_args = parse_args()
    # Make model name visible to the agent via env var
    os.environ["PLANNER_MODEL"] = cli_args.model
    asyncio.run(main(cli_args))

