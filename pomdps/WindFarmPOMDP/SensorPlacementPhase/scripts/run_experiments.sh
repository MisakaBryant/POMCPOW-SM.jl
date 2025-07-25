### Runs script multiple times, with different parameters. ###

# Create new session, detached.
tmux new-session -d -s WindFarm

# Create the windows for each (and close after script exits).    
#                            Activate 15 CPU threads             Generic Runner Script         Runs       Script To Run                     Solver Args
tmux new-window -n:entropy1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_entropy.jl         entropy greedy'
tmux new-window -n:mutuali1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_mutualinfo.jl      mutualinfo greedy'
tmux new-window -n:diffent1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_diffentro.jl       diffentro greedy'

tmux new-window -n:greedyy1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_greedy.jl          greedy greedy'
tmux new-window -n:randomm1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_random.jl          random greedy'

# tmux new-window -n:bayesia1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_bayesian.jl        bayesian greedy 360 20'
tmux new-window -n:genetic1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_genetic.jl         genetic greedy 6 100 0.8 0.05'
tmux new-window -n:pomcpow1 'export JULIA_NUM_THREADS=15;        julia solve_generic.jl        100        solve_SensorPP_pomcpow.jl         pomcpow greedy UCB 500'
