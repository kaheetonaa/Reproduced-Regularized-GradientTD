import numpy as np
import torch
from multiprocessing import Pool,freeze_support
import logging

from RlGlue import RlGlue
from agents.QLearning import QLearning
from agents.QRC import QRC
from agents.QC import QC
from environments.MountainCar import MountainCar

from utils.rl_glue import RlGlueCompatWrapper

ITERATION_NUMBER=0
RUNS = 14#10
EPISODES = 100 #100
LEARNERS = [QRC, QC, QLearning]

data={'QLearning': np.array([-1*np.ones(EPISODES) for i in range(RUNS)]),
    'QRC': np.array([-1*np.ones(EPISODES) for i in range(RUNS)]),
    'QC': np.array([-1* np.ones(EPISODES) for i in range(RUNS)])}

COLORS = {
    'QLearning': 'blue',
    'QRC': 'purple',
    'QC': 'green',
}

# use stepsizes found in parameter study
STEPSIZES = {
    'QLearning': 0.003906,
    'QRC': 0.0009765,
    'QC': 0.0009765,
}

def run_single(args):
    run, Learner = args
    np.random.seed(run+ITERATION_NUMBER)
    torch.manual_seed(run+ITERATION_NUMBER)
    env = MountainCar()

    learner = Learner(env.features, env.num_actions, {
        'alpha': STEPSIZES[Learner.__name__],
        'epsilon': 0.1,
        'beta': 1.0,
        'target_refresh': 1,
        'buffer_size': 4000,
        'h1': 32,
        'h2': 32,
    })

    agent = RlGlueCompatWrapper(learner, gamma=0.99)

    glue = RlGlue(agent, env)

    glue.start()
    results=[]
    for episode in range(EPISODES):
        glue.num_steps = 0
        glue.total_reward = 0
        glue.runEpisode(max_steps=1000)
        results.append(int(glue.num_steps))
        print(Learner.__name__, run, episode, glue.num_steps)
    print(results)
    return results

def main():
	for L in LEARNERS:
		with Pool(processes=14) as pool:
			_d = pool.map(run_single, [(r, L) for r in range(RUNS)])
			data[L.__name__]=np.array(_d[0])
	print(data)
	for key in data.keys():
		np.save(str(ITERATION_NUMBER)+"data"+key+".npy",data[key])

if __name__=="__main__":
	freeze_support()
	main()

#import matplotlib.pyplot as plt
#from utils.plotting import plot

#ax = plt.gca()

#for Learner in LEARNERS:
#    name = Learner.__name__
#    data = collector.getStats(name)
#    plot(ax, data, label=name, color=COLORS[name])

#plt.legend()
#plt.show()
#plt.savefig("fig5.jpg")
