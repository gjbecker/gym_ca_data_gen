import pickle
import os
import sys
import pprint

if len(sys.argv) < 2:
    sys.exit('Missing filename arg!')
else:
    data = sys.argv[1]

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), data)

file = open(path, 'rb')
# print(pickle.load(file))
stats = pickle.load(file)
ep = 0
step = 0
for step in range(4):
    # print(f'Episodes: {len(stats)}')
    # print(f'Ep {ep} Timesteps: {len(stats[ep])}')
    # pprint.pprint(f'Ep {ep}, Step {step}: \n{stats[ep][step]}')
    print(f'State: {stats[ep][step]["states"]}')
    print(f'Action: {stats[ep][step]["actions"]}')