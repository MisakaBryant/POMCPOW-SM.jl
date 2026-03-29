using POMDPs
using BasicPOMCP
using POMDPModels
using POMDPTools
using POMCPOW
using LaserTag
using SubHunt
using Random
using DiscreteValueIteration
using Base.Threads
using Logging, LoggingExtras
using StaticArrays
using LinearAlgebra

include("../pomdps/LunarLander/po_lunar.jl")
include("../pomdps/WindFarmPOMDP/SensorPlacementPhase/src/SensorPP.jl")
include("../pomdps/WindFarmPOMDP/TurbinePlacementPhase/src/TurbinePP.jl")

logger = TeeLogger(
    global_logger(),          # Current global logger (stderr)
    FileLogger("output.log"; append=true) # FileLogger writing to output.log
)

global_logger(logger)

function trail(policy::Policy, pomdp::POMDP; max_steps=100)
    total_reward = 0.0

    # 使用 stepthrough 逐步仿真
    wfparams = WindFieldBeliefParams()
    up = updater(policy)
    b0 = initialize_belief_lookup(wfparams)
    s0 = initialize_state(b0, wfparams)
    steps = hasfield(typeof(pomdp), :timesteps) ? pomdp.timesteps : max_steps
    for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, policy, up, b0, s0, "s,a,r,o,b,t,sp,bp", max_steps=steps)
        total_reward += r * discount(pomdp)^t  # 累加折扣奖励
        if isterminal(pomdp, sp)
            break
        end
    end

    return total_reward
end

function solve_and_evaluate(solver::AbstractPOMCPSolver, pomdp::POMDP; max_steps=100, total=1000)
    policy = solve(solver, pomdp)

    r = Threads.Atomic{Float64}(0.0)

    Threads.@sync for i in 1:total
        Threads.@spawn begin
            success = false
            while !success
                try
                    local_policy = deepcopy(policy)
                    total_reward = trail(local_policy, pomdp, max_steps=max_steps)
                    atomic_add!(r, total_reward)
                    success = true
                catch e
                    # @warn "Error in thread $i: $e"
                    # Base.show_backtrace(stderr, catch_backtrace())
                end
                GC.gc()
            end
        end
    end

    avg_total_reward = r[] / total

    return avg_total_reward
end

pomdp = WindFarmPOMDP()
c = 30
k_a = 3.0
alpha_a = 0.3
k_o = 3.0
alpha_o = 0.3
similarity_threshold = 0.99

rng = MersenneTwister(13)
@show c, k_o, alpha_o, similarity_threshold
tree_queries_list = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
time_limit_list = [0.01, 0.1, 1.0, 5.0, 10.0]

total_reward_list = []

for tree_queries in tree_queries_list
    solver = POMCPOWSolver(tree_queries=tree_queries,
                            criterion=MaxUCB(c),
                            final_criterion=MaxTries(),
                            max_depth=20,
                            max_time=100,
                            # enable_action_pw=true, # 对于离散动作场景
                            enable_action_pw=true,  # |
                            k_action=k_a,           # | 对于连续动作场景
                            alpha_action=alpha_a,   # |
                            k_observation=k_o,
                            alpha_observation=alpha_o,
                            # estimate_value=FOValue(vp),
                            check_repeat_obs=true,
                            # default_action=ReportWhenUsed(-1),
                            rng=rng,
                            similarity_threshold=similarity_threshold    # 观测相似度阈值，check_repeat_obs为true时有效
                            )
    ADR = solve_and_evaluate(solver, pomdp; total=2)
    @info "Total reward with $tree_queries tree queries: ", ADR
    # @info "Total reward with $time_limit s limit: ", ADR
    push!(total_reward_list, ADR)
end