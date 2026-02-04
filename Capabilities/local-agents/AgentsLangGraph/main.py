#!/usr/bin/env python3
"""Main entry point for the LangGraph scientific discovery pipeline."""
import argparse

from pipeline.graph import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-agent scientific discovery pipeline using LangGraph"
    )
    parser.add_argument(
        "--goal", "-g",
        default="Find catalysts that improve CO2 conversion at room temperature.",
        help="The scientific goal to address"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    args = parser.parse_args()

    # Run the pipeline
    final_state = run_pipeline(args.goal, verbose=not args.quiet)

    # Print summary
    print("\n" + "="*60)
    print("Pipeline Complete")
    print("="*60)
    print(f"\nGoal: {final_state['goal']}")
    print(f"\nMessages logged: {len(final_state['messages'])}")
    for msg in final_state['messages']:
        print(f"  - {msg}")


if __name__ == "__main__":
    main()
