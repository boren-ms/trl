#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import fire
from ray_utils import init_ray

def main():
    """Main entry."""
    init_ray()
    print("All done")    

if __name__ == "__main__":
    """Main entry point."""
    fire.Fire(main)