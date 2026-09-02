#!/usr/bin/env python3
"""
CLI entry point for running the 6-model research benchmark.
Usage:
  python3 run_benchmark.py --models all --images 5 --seed 42
"""
import sys
from benchmark import main

if __name__ == "__main__":
    main()
