import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('alpha', type=int, default=1, nargs='?', help='first term')
parser.add_argument('-b', '--beta', type=int, default=2, nargs='?')
parser.add_argument('-g', '--gamma', type=int, default=3, nargs='?')
parser.add_argument('delta', type=int, default=4, nargs='?')
parser.add_argument('--sigma', action='store_true')
namespace = parser.parse_args()

print(
    f'alpha={namespace.alpha}, '
    f'beta={namespace.beta}, '
    f'gamma={namespace.gamma}, '
    f'delta={namespace.delta}, '
    f'sigma={namespace.sigma}'
)
