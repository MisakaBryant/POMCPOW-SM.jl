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

pomdp = WindFarmPOMDP()
c = 1
k_a = 3.0
alpha_a = 0.3
k_o = 3.0
alpha_o = 0.3
similarity_threshold = 0.99

tree_queries = 100
rng = MersenneTwister(13)

solver = POMCPOWSolver(tree_queries=tree_queries,
                            criterion=MaxUCB(c),
                            final_criterion=MaxTries(),
                            max_depth=20,
                            max_time=100,
                            # enable_action_pw=false, # 对于离散动作场景
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

                            
function trail(policy::Policy, pomdp::POMDP; max_steps=100)
    total_reward = 0.0

    # 使用 stepthrough 逐步仿真
    wfparams = WindFieldBeliefParams()
    up = updater(policy)
    b0 = initialize_belief_lookup(wfparams)
    s0 = initialize_state(b0, wfparams)
    steps = hasfield(typeof(pomdp), :timesteps) ? pomdp.timesteps : max_steps
    @show pomdp, policy, max_steps
    for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, policy, up, b0, s0, "s,a,r,o,b,t,sp,bp", max_steps=steps)
        @show t, a, r
        total_reward += r * discount(pomdp)^t  # 累加折扣奖励
        if isterminal(pomdp, sp)
            break
        end
    end

    return total_reward
end

# function trail(policy::Policy, pomdp::POMDP; max_steps=100)
#     steps = hasfield(typeof(pomdp), :timesteps) ? pomdp.timesteps : max_steps
#     hr = HistoryRecorder(max_steps=steps)
#     rhist = simulate(hr, pomdp, policy)
#     # for (s, b, a, r, sp, o) in rhist
#     #     @show s, a, r, sp
#     # end

#     return discounted_reward(rhist)
# end

@show r = trail(solve(solver, pomdp), pomdp)