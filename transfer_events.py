import argparse
from Analysis.load_data import events_to_BIDS_dir

parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir')
parser.add_argument('--subjs', nargs="+")
args = parser.parse_args()
subjs = args.subjs
bids_dir = args.bids_dir

for subj in subjs:
    events_to_BIDS_dir(subj, bids_dir)