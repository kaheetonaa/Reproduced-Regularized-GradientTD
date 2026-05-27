import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import numpy as np

    # source found at: https://github.com/andnp/RlGlue
    from RlGlue import RlGlue
    from utils.Collector import Collector
    from utils.policies import actionArrayToPolicy, matrixToPolicy
    from utils.rl_glue import RlGlueCompatWrapper
    from utils.errors import buildRMSPBE

    from environments.RandomWalk import RandomWalk, TabularRep, DependentRep, InvertedRep
    from environments.Boyan import Boyan, BoyanRep
    from environments.Baird import Baird, BairdRep

    from agents.TD import TD
    from agents.TDC import TDC
    from agents.HTD import HTD
    from agents.GTD2 import GTD2
    from agents.TDRC import TDRC
    from agents.Vtrace import Vtrace



    return (
        Baird,
        BairdRep,
        Boyan,
        BoyanRep,
        Collector,
        DependentRep,
        GTD2,
        HTD,
        InvertedRep,
        RandomWalk,
        RlGlue,
        RlGlueCompatWrapper,
        TD,
        TDC,
        TDRC,
        TabularRep,
        Vtrace,
        actionArrayToPolicy,
        buildRMSPBE,
        matrixToPolicy,
        np,
    )


@app.cell
def _(
    Baird,
    BairdRep,
    Boyan,
    BoyanRep,
    DependentRep,
    GTD2,
    HTD,
    InvertedRep,
    RandomWalk,
    TD,
    TDC,
    TDRC,
    TabularRep,
    Vtrace,
    actionArrayToPolicy,
    matrixToPolicy,
    np,
):
    # --------------------------------
    # Set up parameters for experiment
    # --------------------------------

    RUNS = 10
    LEARNERS = [GTD2, TDC, Vtrace, HTD, TD, TDRC]

    PROBLEMS = [
        # 5-state random walk environment with tabular features
        {
            'env': RandomWalk,
            'representation': TabularRep,
            # go LEFT 40% of the time
            'target': actionArrayToPolicy([0.4, 0.6]),
            # take each action equally
            'behavior': actionArrayToPolicy([0.5, 0.5]),
            'gamma': 1.0,
            'steps': 3000,
            # hardcode stepsizes found from parameter study
            'stepsizes': {
                'TD': 0.03125,
                'TDRC': 0.03125,
                'TDC': 0.0625,
                'GTD2': 0.03125,
                'HTD': 0.03125,
                'Vtrace': 0.03125,
            }
        },
        # 5-state random walk environment with dependent features
        {
            'env': RandomWalk,
            'representation': DependentRep,
            # go LEFT 40% of the time
            'target': actionArrayToPolicy([0.4, 0.6]),
            # take each action equally
            'behavior': actionArrayToPolicy([0.5, 0.5]),
            'gamma': 1.0,
            'steps': 3000,
            # hardcode stepsizes found from parameter study
            'stepsizes': {
                'TD': 0.03125,
                'TDRC': 0.03125,
                'TDC': 0.0625,
                'GTD2': 0.0625,
                'HTD': 0.03125,
                'Vtrace': 0.03125,
            }
        },
        # 5-state random walk environment with inverted features
        {
            'env': RandomWalk,
            'representation': InvertedRep,
            # go LEFT 40% of the time
            'target': actionArrayToPolicy([0.4, 0.6]),
            # take each action equally
            'behavior': actionArrayToPolicy([0.5, 0.5]),
            'gamma': 1.0,
            'steps': 3000,
            # hardcode stepsizes found from parameter study
            'stepsizes': {
                'TD': 0.125,
                'TDRC': 0.125,
                'TDC': 0.125,
                'GTD2': 0.125,
                'HTD': 0.125,
                'Vtrace': 0.125,
            }
        },
        # Boyan's chain
        {
            'env': Boyan,
            'representation': BoyanRep,
            # go LEFT 40% of the time
            'target': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
            # take each action equally
            'behavior': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
            'gamma': 1.0,
            'steps': 3000, # H: change steps to 3000 from 10000
            # hardcode stepsizes found from parameter study
            'stepsizes': {
                'TD': 0.0625,
                'TDRC': 0.0625,
                'TDC': 0.5,
                'GTD2': 0.5,
                'HTD': 0.0625,
                'Vtrace': 0.0625,
            }
        },
        # Baird's Counter-example domain
        {
            'env': Baird,
            'representation': BairdRep,
            # go LEFT 40% of the time
            'target': actionArrayToPolicy([0., 1.]),
            # take each action equally
            'behavior': actionArrayToPolicy([6/7, 1/7]),
            'starting_condition': np.array([1, 1, 1, 1, 1, 1, 1, 10]),
            'gamma': 0.99,
            'steps': 3000,  # H: change steps to 3000 from 20000
            # hardcode stepsizes found from parameter study
            'stepsizes': {
                'TD': 0.00390625,  #[:-1] # H: change step sizes to same as TDRC
                'TDRC': 0.00390625, #[:-1] # H: change step sizes to same as all
                'TDC': 0.00390625,
                'GTD2': 0.00390625,
                'HTD': 0.00390625,
                'Vtrace': 0.00390625,
            }
        },
    ]

    COLORS = {
        'TD': 'lightgray',
        'TDC': 'lightgray',
        'TDRC': 'pink',
        'GTD2': 'lightgray',
        'Vtrace': 'lightgray',
        'HTD': 'lightgray',
    } # H: change color

    PROBLEMS=PROBLEMS #[:-1] # H:remove baird problem
    PROBLEMS
    return COLORS, LEARNERS, PROBLEMS, RUNS


@app.cell
def _(
    Collector,
    LEARNERS,
    PROBLEMS,
    RUNS,
    RlGlue,
    RlGlueCompatWrapper,
    buildRMSPBE,
    np,
):
    # -----------------------------------
    # Collect the data for the experiment
    # -----------------------------------

    # a convenience object to store data collected during runs
    collector = Collector()

    for run in range(RUNS):
        for problem in PROBLEMS:
            for Learner in LEARNERS:
                # for reproducibility, set the random seed for each run
                # also reset the seed for each learner, so we guarantee each sees the same data
                np.random.seed(run)

                # build a new instance of the environment each time
                # just to be sure we don't bleed one learner into the next
                Env = problem['env']
                env = Env()

                target = problem['target']
                behavior = problem['behavior']

                Rep = problem['representation']
                rep = Rep()

                print(run, Env.__name__, Rep.__name__, Learner.__name__)

                # build the X, P, R, and D matrices for computing RMSPBE
                X, P, R, D = env.getXPRD(target, rep)
                RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

                # build a new instance of the learning algorithm
                learner = Learner(rep.features(), {
                    'gamma': problem['gamma'],
                    'alpha': problem['stepsizes'][Learner.__name__],
                    'beta': 1,
                })

                # build an "agent" which selects actions according to the behavior
                # and tries to estimate according to the target policy
                agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)

                # for Baird's counter-example, set the initial value function manually
                if problem.get('starting_condition') is not None:
                    learner.w = problem['starting_condition'].copy()

                # build the experiment runner
                # ties together the agent and environment
                # and allows executing the agent-environment interface from Sutton-Barto
                glue = RlGlue(agent, env)

                # start the episode (env produces a state then agent produces an action)
                glue.start()
                for step in range(problem['steps']):
                    # interface sends action to env and produces a next-state and reward
                    # then sends the next-state and reward to the agent to make an update
                    _, _, _, terminal = glue.step()

                    # when we hit a terminal state, start a new episode
                    if terminal:
                        glue.start()

                    # evaluate the RMPSBE
                    # subsample to reduce computational cost
                    if step % 100 == 0: # H: change to average 200 time from 100 time
                        w = learner.getWeights()
                        rmspbe = RMSPBE(w)

                        #  create a unique key to store the data for this env/representation/agent tuple
                        data_key = f'{Env.__name__}-{Rep.__name__}-{Learner.__name__}'
                        # store the data in the "collector" until we need it for plotting
                        collector.collect(data_key, rmspbe)

                # tell the data collector we're done collecting data for this env/learner/rep combination
                collector.reset()

    return collector, rmspbe


@app.cell
def _(rmspbe):
    rmspbe
    return


@app.cell
def _(collector):
    collector.all_data.keys()
    return


@app.cell
def _(COLORS, LEARNERS, PROBLEMS, collector):
    def plot_compare():
        # ---------------------
        # Plotting the bar plot
        # ---------------------
        import matplotlib.pyplot as plt

        ax = plt.gca()
        f = plt.gcf()

        # get TDRC's baseline performance for each problem
        baselines = [None] * len(PROBLEMS)
        for i, problem in enumerate(PROBLEMS):
            env = problem['env'].__name__
            rep = problem['representation'].__name__

            mean_curve, _, _ = collector.getStats(f'{env}-{rep}-TDRC')

            # compute TDRC's AUC
            baselines[i] = mean_curve.mean()

        # how far from the left side of the plot to put the bar
        offset = -3
        for i, problem in enumerate(PROBLEMS):
            # additional offset between problems
            # creates space between the problems
            offset += 3
            for j, Learner in enumerate(LEARNERS):
                learner = Learner.__name__
                env = problem['env'].__name__
                rep = problem['representation'].__name__

                x = i * len(LEARNERS) + j + offset

                mean_curve, stderr_curve, runs = collector.getStats(f'{env}-{rep}-{learner}')
                auc = mean_curve.mean()
                auc_stderr = stderr_curve.mean()

                relative_auc = auc / baselines[i]
                relative_stderr = auc_stderr / baselines[i]

                ax.bar(x, relative_auc, yerr=relative_stderr, color=COLORS[learner], tick_label='')
                ax.set_ylim([0,3])
        return plt.show()


    plot_compare()
    return


@app.cell
def _(
    Collector,
    LEARNERS,
    PROBLEMS,
    RUNS,
    RlGlue,
    RlGlueCompatWrapper,
    buildRMSPBE,
    np,
):
    import math
    def alpha_sensitivity(_range_min,_range_max,_max):

        # -----------------------------------
        # Collect the data for the experiment
        # -----------------------------------

        # a convenience object to store data collected during runs
        # H: update alpha sensitivity in _range from 0 to 5, return collectors with key=alpha, value=collector   
        collector = Collector()
        for alpha_power in range(_range_min,_range_max):
            alpha=2**(-alpha_power)

            for run in range(RUNS):
                for problem in PROBLEMS:
                    for Learner in LEARNERS:

                        problem['stepsizes'][Learner.__name__]=alpha
                        # for reproducibility, set the random seed for each run
                        # also reset the seed for each learner, so we guarantee each sees the same data
                        np.random.seed(run)

                        # build a new instance of the environment each time
                        # just to be sure we don't bleed one learner into the next
                        Env = problem['env']
                        env = Env()

                        target = problem['target']
                        behavior = problem['behavior']

                        Rep = problem['representation']
                        rep = Rep()

                        print(run, Env.__name__, Rep.__name__, Learner.__name__)

                        # build the X, P, R, and D matrices for computing RMSPBE
                        X, P, R, D = env.getXPRD(target, rep)
                        RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

                        # build a new instance of the learning algorithm
                        learner = Learner(rep.features(), {
                            'gamma': problem['gamma'],
                            'alpha': alpha,
                            'beta': 1,
                        })

                        # build an "agent" which selects actions according to the behavior
                        # and tries to estimate according to the target policy
                        agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)

                        # for Baird's counter-example, set the initial value function manually
                        if problem.get('starting_condition') is not None:
                            learner.w = problem['starting_condition'].copy()

                        # build the experiment runner
                        # ties together the agent and environment
                        # and allows executing the agent-environment interface from Sutton-Barto
                        glue = RlGlue(agent, env)

                        # start the episode (env produces a state then agent produces an action)
                        glue.start()
                        for step in range(problem['steps']):
                            # interface sends action to env and produces a next-state and reward
                            # then sends the next-state and reward to the agent to make an update
                            _, _, _, terminal = glue.step()

                            # when we hit a terminal state, start a new episode
                            if terminal:
                                glue.start()

                            # evaluate the RMPSBE
                            # subsample to reduce computational cost
                            if step % 100 == 0: 
                                w = learner.getWeights()
                                rmspbe = RMSPBE(w)
                                if math.isinf(rmspbe) or math.isnan(rmspbe): #rmspbe>_max
                                    rmspbe=_max
                                    terminal=True
                                #  create a unique key to store the data for this env/representation/agent tuple
                                data_key = f'{alpha_power}-{Env.__name__}-{Rep.__name__}-{Learner.__name__}'
                                # store the data in the "collector" until we need it for plotting
                                collector.collect(data_key, rmspbe)

                        # tell the data collector we're done collecting data for this env/learner/rep combination
                        collector.reset()
        return collector


    alpha_sensitivity=alpha_sensitivity(0,8,9e+200)
    return alpha_sensitivity, math


@app.cell
def _(LEARNERS, PROBLEMS):
    import pandas as pd

    def plot_compare_sensitivity(collector,_range_min,_range_max,top_lim,bot_lim):
        # ---------------------
        # Plotting the bar plot
        # ---------------------
        import matplotlib.pyplot as plt

        # get TDRC's baseline performance for each problem
        baselines = [None] * len(PROBLEMS) * (_range_max-_range_min)
        for alpha_power in range(_range_min,_range_max):

            for i, problem in enumerate(PROBLEMS):
                env = problem['env'].__name__
                rep = problem['representation'].__name__

                mean_curve, _, _ = collector.getStats(f'{alpha_power}-{env}-{rep}-TDRC')

                # compute TDRC's AUC
                baselines[alpha_power*len(PROBLEMS)+i] = mean_curve.mean()
        print(baselines)


        #H: DF index =-alpha_power, learner: columns, 
        _rep=[i["representation"].__name__ for i in PROBLEMS]
        _lea=[i.__name__ for i in LEARNERS]
        data={i:{j:[] for j in _lea} for i in _rep}


        for alpha_power in range(_range_min,_range_max):
            for i, problem in enumerate(PROBLEMS):
                # additional offset between problems
                # creates space between the problems
                for j, Learner in enumerate(LEARNERS):
                    learner = Learner.__name__
                    env = problem['env'].__name__
                    rep = problem['representation'].__name__

                    x = -alpha_power

                    mean_curve, stderr_curve, runs = collector.getStats(f'{alpha_power}-{env}-{rep}-{learner}')
                    auc = mean_curve.mean()
                    auc_stderr = stderr_curve.mean()

                    relative_auc = auc / baselines[alpha_power * len(PROBLEMS)+i]
                    relative_stderr = auc_stderr / baselines[alpha_power * len(PROBLEMS)+i]
                    data[rep][learner]+={auc}

        #return plt.show()
        for _r in _rep:
            df=pd.DataFrame(data[_r])
            plot=df.plot(title=_r)
            plot.set_ylim(top=top_lim,bottom=bot_lim) 
        return plt.show()



    return (plot_compare_sensitivity,)


@app.cell
def _(alpha_sensitivity, plot_compare_sensitivity):
    plot_compare_sensitivity(alpha_sensitivity,0,8,top_lim=2,bot_lim=0)
    return


@app.cell
def _(
    Collector,
    GTD2,
    HTD,
    LEARNERS,
    PROBLEMS,
    RUNS,
    RlGlue,
    RlGlueCompatWrapper,
    TDRC,
    buildRMSPBE,
    math,
    np,
):
    def eta_sensitivity(_range_min,_range_max,_max):

        # -----------------------------------
        # Collect the data for the experiment
        # -----------------------------------

        # a convenience object to store data collected during runs
        # H: update alpha sensitivity in _range from 0 to 5, return collectors with key=alpha, value=collector   
        collector = Collector()
        for eta in range(_range_min,_range_max):
            for run in range(RUNS):
                for problem in PROBLEMS:
                    for Learner in LEARNERS:

                        # for reproducibility, set the random seed for each run
                        # also reset the seed for each learner, so we guarantee each sees the same data
                        np.random.seed(run)

                        # build a new instance of the environment each time
                        # just to be sure we don't bleed one learner into the next
                        Env = problem['env']
                        env = Env()

                        target = problem['target']
                        behavior = problem['behavior']

                        Rep = problem['representation']
                        rep = Rep()

                        print(run, Env.__name__, Rep.__name__, Learner.__name__)

                        # build the X, P, R, and D matrices for computing RMSPBE
                        X, P, R, D = env.getXPRD(target, rep)
                        RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

                        # build a new instance of the learning algorithm
                        if Learner in [HTD,GTD2,TDRC]:
                            learner = Learner(rep.features(), {
                            'gamma': problem['gamma'],
                            'alpha': problem['stepsizes'][Learner.__name__],
                            'beta': 1,
                            "eta":eta
                            })
                        else:
                                learner = Learner(rep.features(), {
                                    'gamma': problem['gamma'],
                                    'alpha': problem['stepsizes'][Learner.__name__],
                                    'beta': 1,
                                })

                        # build an "agent" which selects actions according to the behavior
                        # and tries to estimate according to the target policy
                        agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)

                        # for Baird's counter-example, set the initial value function manually
                        if problem.get('starting_condition') is not None:
                            learner.w = problem['starting_condition'].copy()

                        # build the experiment runner
                        # ties together the agent and environment
                        # and allows executing the agent-environment interface from Sutton-Barto
                        glue = RlGlue(agent, env)

                        # start the episode (env produces a state then agent produces an action)
                        glue.start()
                        for step in range(problem['steps']):
                            # interface sends action to env and produces a next-state and reward
                            # then sends the next-state and reward to the agent to make an update
                            _, _, _, terminal = glue.step()

                            # when we hit a terminal state, start a new episode
                            if terminal:
                                glue.start()

                            # evaluate the RMPSBE
                            # subsample to reduce computational cost
                            if step % 100 == 0: 
                                w = learner.getWeights()
                                rmspbe = RMSPBE(w)
                                if math.isinf(rmspbe) or math.isnan(rmspbe): #rmspbe>_max
                                    rmspbe=_max
                                    terminal=True
                                #  create a unique key to store the data for this env/representation/agent tuple
                                data_key = f'{eta}-{Env.__name__}-{Rep.__name__}-{Learner.__name__}'
                                # store the data in the "collector" until we need it for plotting
                                collector.collect(data_key, rmspbe)

                        # tell the data collector we're done collecting data for this env/learner/rep combination
                        collector.reset()
        return collector


    eta_sensitivity=eta_sensitivity(-8,8,9e+200)
    return (eta_sensitivity,)


@app.cell
def _(eta_sensitivity, plot_compare_sensitivity):
    plot_compare_sensitivity(eta_sensitivity,-8,8,top_lim=1,bot_lim=0)
    return


@app.cell
def _(
    Collector,
    LEARNERS,
    PROBLEMS,
    RUNS,
    RlGlue,
    RlGlueCompatWrapper,
    buildRMSPBE,
    eta,
    math,
    np,
):
    def beta_sensitivity(_range_min,_range_max,_max):

        # -----------------------------------
        # Collect the data for the experiment
        # -----------------------------------

        # a convenience object to store data collected during runs
        # H: update alpha sensitivity in _range from 0 to 5, return collectors with key=alpha, value=collector   
        collector = Collector()
        for beta in range(_range_min,_range_max):
            for run in range(RUNS):
                for problem in PROBLEMS:
                    for Learner in LEARNERS:

                        # for reproducibility, set the random seed for each run
                        # also reset the seed for each learner, so we guarantee each sees the same data
                        np.random.seed(run)

                        # build a new instance of the environment each time
                        # just to be sure we don't bleed one learner into the next
                        Env = problem['env']
                        env = Env()

                        target = problem['target']
                        behavior = problem['behavior']

                        Rep = problem['representation']
                        rep = Rep()

                        print(run, Env.__name__, Rep.__name__, Learner.__name__)

                        # build the X, P, R, and D matrices for computing RMSPBE
                        X, P, R, D = env.getXPRD(target, rep)
                        RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])

                        # build a new instance of the learning algorithm
                        learner = Learner(rep.features(), {
                            'gamma': problem['gamma'],
                            'alpha': problem['stepsizes'][Learner.__name__],
                            'beta': beta,
                        })

                        # build an "agent" which selects actions according to the behavior
                        # and tries to estimate according to the target policy
                        agent = RlGlueCompatWrapper(learner, behavior, target, rep.encode)

                        # for Baird's counter-example, set the initial value function manually
                        if problem.get('starting_condition') is not None:
                            learner.w = problem['starting_condition'].copy()

                        # build the experiment runner
                        # ties together the agent and environment
                        # and allows executing the agent-environment interface from Sutton-Barto
                        glue = RlGlue(agent, env)

                        # start the episode (env produces a state then agent produces an action)
                        glue.start()
                        for step in range(problem['steps']):
                            # interface sends action to env and produces a next-state and reward
                            # then sends the next-state and reward to the agent to make an update
                            _, _, _, terminal = glue.step()

                            # when we hit a terminal state, start a new episode
                            if terminal:
                                glue.start()

                            # evaluate the RMPSBE
                            # subsample to reduce computational cost
                            if step % 100 == 0: 
                                w = learner.getWeights()
                                rmspbe = RMSPBE(w)
                                if math.isinf(rmspbe) or math.isnan(rmspbe): #rmspbe>_max
                                    rmspbe=_max
                                    terminal=True
                                #  create a unique key to store the data for this env/representation/agent tuple
                                data_key = f'{eta}-{Env.__name__}-{Rep.__name__}-{Learner.__name__}'
                                # store the data in the "collector" until we need it for plotting
                                collector.collect(data_key, rmspbe)

                        # tell the data collector we're done collecting data for this env/learner/rep combination
                        collector.reset()
        return collector


    beta_sensitivity=beta_sensitivity(0,8,9e+200)
    return (beta_sensitivity,)


@app.cell
def _(beta_sensitivity, plot_compare_sensitivity):
    plot_compare_sensitivity(beta_sensitivity,0,8,top_lim=1,bot_lim=0)
    return


if __name__ == "__main__":
    app.run()
