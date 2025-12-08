"""CLI main for b3_2d."""

import argparse
import sys
from b3_2d.core.mesh import process_vtp_multi_section


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Process VTP files for b3_2d meshing.")
    parser.add_argument("-f", "--file", required=True, help="Path to the VTP file.")
    parser.add_argument("-o", "--output", required=True, help="Output base directory.")
    parser.add_argument(
        "-p", "--processes", type=int, default=None, help="Number of processes to use."
    )
    args = parser.parse_args()
    try:
        process_vtp_multi_section(args.file, args.output, args.processes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
