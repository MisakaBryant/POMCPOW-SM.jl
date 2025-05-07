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


logger = TeeLogger(
    global_logger(),          # Current global logger (stderr)
    FileLogger("output.log"; append=true) # FileLogger writing to output.log
)

global_logger(logger)

# const Vec8 = SVector{8, Float64}

# function POMCPOW.obsSimilar(::SubHuntPOMDP, o1::Vec8, o2::Vec8, similarity_threshold::Float64)
#     return exp(-norm(o1 - o2)^2) >= similarity_threshold
# end

# function POMCPOW.simplifyObs(::SubHuntPOMDP, o::Vec8, step::Float64=0.1)
#     return Vec8(round(x/step)*step for x in o)
# end

function trail(policy::Policy, pomdp::POMDP; max_steps=100)
    hr = HistoryRecorder(max_steps=max_steps)
    rhist = simulate(hr, pomdp, policy)
    # for (s, b, a, r, sp, o) in rhist
    #     @show s, a, r, sp
    # end

    return discounted_reward(rhist)
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
            end
        end
    end

    avg_total_reward = r[] / total

    return avg_total_reward
end

pomdp = LightDark1D()
c = 90
k_o = 5
alpha_o = 1/15
similarity_threshold = 0.99

# pomdp = SubHuntPOMDP()
# c = 17
# k_o = 6
# alpha_o = 1/100
# similarity_threshold = 0.99

# pomdp = gen_lasertag(rng=MersenneTwister(7), robot_position_known=true)
# c = 26 
# k_o = 4 
# alpha_o = 1/35
# similarity_threshold = 0.99

# pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))
# c = 110
# k_a = 30
# alpha_a = 1/30
# k_o = 5
# alpha_o = 1/100

# LightDark1D 需要注释掉
# vs = ValueIterationSolver()
# if !isdefined(Main, :vp) || vp.mdp != pomdp
#     mdp = UnderlyingMDP(pomdp)
#     vp = solve(vs, mdp)
# end
rng = MersenneTwister(13)
@show c, k_o, alpha_o, similarity_threshold
tree_queries_list = [10000, 20000, 50000, 100000, 200000]
time_limit_list = [0.01, 0.1, 1.0, 5.0, 10.0]

total_reward_list = []

for tree_queries in tree_queries_list
    solver = POMCPOWSolver(tree_queries=tree_queries,
                            criterion=MaxUCB(c),
                            final_criterion=MaxTries(),
                            max_depth=20,
                            max_time=100,
                            enable_action_pw=true, # 对于离散动作场景
                            # enable_action_pw=true,  # |
                            # k_action=k_a,           # | 对于连续动作场景
                            # alpha_action=alpha_a,   # |
                            k_observation=k_o,
                            alpha_observation=alpha_o,
                            # estimate_value=FOValue(vp),
                            check_repeat_obs=true,
                            # default_action=ReportWhenUsed(-1),
                            rng=rng,
                            similarity_threshold=similarity_threshold    # 观测相似度阈值，check_repeat_obs为true时有效
                            )
    ADR = solve_and_evaluate(solver, pomdp)
    @info "Total reward with $tree_queries tree queries: ", ADR
    # @info "Total reward with $time_limit s limit: ", ADR
    push!(total_reward_list, ADR)
end

# p = plot(tree_queries_list, total_reward_list, label="Total reward", xlabel="Tree queries", ylabel="Total reward", title="Total reward with different tree queries")

# png(p, "POMCPOW_SubHunt")