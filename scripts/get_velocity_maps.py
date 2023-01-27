import yaml
import sys
import os

config_file = sys.argv[1]  # Input parameter file as first argument
with open(config_file, 'r') as file:
    params = yaml.safe_load(file)

