import argparse
parser=argparse.ArgumentParser(description="fuck")
parser.add_argument('--seq_len', default=10, type=int, choices=[5,10,20])
parser.add_argument('--load', default=False, action='store_true')
args=vars(parser.parse_args())
print(args)
