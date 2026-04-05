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
using Statistics
using Printf
using Plots

include("../pomdps/LunarLander/po_lunar.jl")
include("../pomdps/WindFarmPOMDP/SensorPlacementPhase/src/SensorPP.jl")

logger = TeeLogger(
    global_logger(),          # Current global logger (stderr)
    FileLogger("output.log"; append=true) # FileLogger writing to output.log
)

global_logger(logger)

# function trail(policy::Policy, pomdp::POMDP; max_steps=100)
#     hr = HistoryRecorder(max_steps=max_steps)
#     rhist = simulate(hr, pomdp, policy)
#     # for (s, b, a, r, sp, o) in rhist
#     #     @show s, a, r, sp
#     # end

#     return discounted_reward(rhist)
# end

function trail(policy::Policy, pomdp::POMDP; max_steps=100)
    total_reward = 0.0

    # 使用 stepthrough 逐步仿真
    for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, policy, "s,a,r,o,b,t,sp,bp", max_steps=max_steps)
        total_reward += r * discount(pomdp)^t  # 累加折扣奖励
        if isterminal(pomdp, sp)
            break
        end
    end

    return total_reward
end

function summarize_rewards(rewards::Vector{Float64})
    n = length(rewards)
    sample_std = n > 1 ? std(rewards; corrected=true) : 0.0
    return (
        n=n,
        min=minimum(rewards),
        max=maximum(rewards),
        mean=mean(rewards),
        var=n > 1 ? var(rewards; corrected=true) : 0.0,
        std=sample_std,
        se=n > 0 ? sample_std / sqrt(n) : 0.0
    )
end

function solve_and_evaluate(solver, pomdp::POMDP; max_steps=100, total=1000, max_retries=5)
    policy = solve(solver, pomdp)
    rewards = fill(NaN, total)

    # -------------
    Threads.@sync for i in 1:total
        Threads.@spawn begin
    # -------------
    # for i in 1:total
    #     begin
    # -------------
            success = false
            tries = 0
            while !success && tries < max_retries
                # try
                    tries += 1
                    local_policy = deepcopy(policy)
                    total_reward = trail(local_policy, pomdp, max_steps=max_steps)
                    rewards[i] = total_reward
                    success = true
                # catch e
                #     @warn "Error in thread $i: $e"
                #     Base.show_backtrace(stderr, catch_backtrace())
                # end
            end
            if !success
                @warn "Trial $i failed after $max_retries retries."
            end
        end
    end

    valid_rewards = collect(filter(!isnan, rewards))
    isempty(valid_rewards) && error("All trials failed. No valid reward collected.")

    return summarize_rewards(valid_rewards), valid_rewards
end

function parse_solver_modes(args::Vector{String})
    supported = Set(["POMCPOW", "SM"])
    if isempty(args)
        return ["POMCPOW", "SM"]
    end

    parsed = String[]
    for arg in args
        for token in split(arg, ',')
            mode = uppercase(strip(token))
            isempty(mode) && continue
            if mode in supported
                push!(parsed, mode)
            elseif mode == "IPFT"
                error("IPFT mode has been removed from runner2.jl. Please run .\\runner\\ipft_runner.jl for IPFT experiments.")
            else
                error("Unknown solver mode: $mode. Supported modes are POMCPOW, SM.")
            end
        end
    end

    isempty(parsed) && error("No valid solver mode provided. Use POMCPOW or SM.")
    return unique(parsed)
end

# pomdp = LightDark1D()
# c = 90
# k_o = 5
# alpha_o = 1/15
# similarity_threshold = 0.99
# tree_queries_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

pomdp = SubHuntPOMDP()
c = 17
k_o = 6.0
alpha_o = 1/100
similarity_threshold = 0.99
tree_queries_list = [5000, 10000, 20000, 50000, 80000]

# pomdp = gen_lasertag(rng=MersenneTwister(7), robot_position_known=true)
# c = 26 
# k_o = 4 
# alpha_o = 1/35
# similarity_threshold = 0.99
# tree_queries_list = [5000, 10000, 20000, 50000]

# pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))
# c = 110
# k_a = 30
# alpha_a = 1/30
# k_o = 5
# alpha_o = 1/100

# pomdp = LunarLander()
# c = 1
# k_o = 10
# alpha_o = 0.5
# k_a = 10
# alpha_a = 0.5
# similarity_threshold = 0.99
# tree_queries_list = [500, 1000, 2000, 3000, 5000, 8000, 10000]

# pomdp = WindFarmPOMDP()
# c = 1
# k_a = 3.0
# alpha_a = 0.3
# k_o = 3.0
# alpha_o = 0.3
# similarity_threshold = 0.99

# LightDark1D 需要注释掉
vs = ValueIterationSolver()
if !isdefined(Main, :vp) || vp.mdp != pomdp
    mdp = UnderlyingMDP(pomdp)
    vp = solve(vs, mdp)
end

rng = MersenneTwister(13)
@show c, k_o, alpha_o, similarity_threshold
time_limit_list = [0.01, 0.1, 1.0, 5.0, 10.0]
total = 1000

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

selected_solver_modes = parse_solver_modes(ARGS)
@info "Selected solver modes=$(join(selected_solver_modes, ","))"

for solver_mode in selected_solver_modes
    stats_table = NamedTuple[]
    all_rewards_by_query = Dict{Int, Vector{Float64}}()
    algo_name = solver_mode
    algo_rng = MersenneTwister(13)

    @info "Running experiment with algorithm=$algo_name"

    for tree_queries in tree_queries_list
        solver = if solver_mode == "POMCPOW"
            POMCPOWSolver(tree_queries=tree_queries,
                        criterion=MaxUCB(c),
                        final_criterion=MaxTries(),
                        max_depth=20,
                        max_time=100,
                        enable_action_pw=false, # 对于离散动作场景
                        # enable_action_pw=true,  # |
                        # k_action=k_a,           # | 对于连续动作场景
                        # alpha_action=alpha_a,   # |
                        k_observation=k_o,
                        alpha_observation=alpha_o,
                        estimate_value=FOValue(vp),
                        check_repeat_obs=true,
                        rng=algo_rng,
                        similarity_threshold=similarity_threshold,
                        algorithm=SOLVER2)
        elseif solver_mode == "SM"
            POMCPOWSolver(tree_queries=tree_queries,
                        criterion=MaxUCB(c),
                        final_criterion=MaxTries(),
                        max_depth=20,
                        max_time=100,
                        enable_action_pw=false,  # 对于离散动作场景
                        # enable_action_pw=true,  # |
                        # k_action=k_a,           # | 对于连续动作场景
                        # alpha_action=alpha_a,   # |
                        k_observation=k_o,
                        alpha_observation=alpha_o,
                        estimate_value=FOValue(vp),
                        check_repeat_obs=true,
                        rng=algo_rng,
                        similarity_threshold=similarity_threshold,
                        algorithm=SOLVER4)
        else
            error("Unsupported solver mode: $solver_mode")
        end

        stats = nothing
        rewards = Float64[]
        stats, rewards = solve_and_evaluate(solver, pomdp, total=total)


        @info @sprintf("algorithm=%s, tree_queries=%d, n=%d, min=%.4f, max=%.4f, mean=%.4f, var=%.4f, std=%.4f, se=%.4f",
                       algo_name, tree_queries, stats.n, stats.min, stats.max, stats.mean, stats.var, stats.std, stats.se)
        push!(stats_table, (
            tree_queries=tree_queries,
            n=stats.n,
            min=stats.min,
            max=stats.max,
            mean=stats.mean,
            var=stats.var,
            std=stats.std,
            se=stats.se
        ))
        all_rewards_by_query[tree_queries] = rewards
    end

    if isempty(stats_table)
        @warn "No valid results for algorithm=$algo_name, skip CSV/plot generation."
        continue
    end

    stats_csv_path = joinpath(results_dir, "reward_stats_by_tree_queries_$(algo_name).csv")
    open(stats_csv_path, "w") do io
        write(io, "tree_queries,n,min,max,mean,var,std,se\n")
        for row in stats_table
            write(io, @sprintf("%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f\n",
                               row.tree_queries, row.n, row.min, row.max, row.mean, row.var, row.std, row.se))
        end
    end

    valid_queries = [row.tree_queries for row in stats_table]
    mean_list = [row.mean for row in stats_table]
    se_list = [row.se for row in stats_table]
    min_list = [row.min for row in stats_table]
    max_list = [row.max for row in stats_table]

    p_summary = plot(valid_queries, mean_list,
                     yerror=se_list,
                     marker=:circle,
                     linewidth=2,
                     label="Mean ± SE",
                     xlabel="Tree queries",
                     ylabel="Discounted reward",
                     title="[$(algo_name)] Reward statistics over $(total) trials")
    scatter!(p_summary, valid_queries, min_list, marker=:diamond, label="Min")
    scatter!(p_summary, valid_queries, max_list, marker=:utriangle, label="Max")

    summary_plot_path = joinpath(results_dir, "reward_summary_vs_tree_queries_$(algo_name).png")
    savefig(p_summary, summary_plot_path)

    hist_plots = [histogram(all_rewards_by_query[q],
                            bins=30,
                            xlabel="Reward",
                            ylabel="Count",
                            title="[$(algo_name)] tree_queries=$q",
                            legend=false) for q in valid_queries]

    rows = ceil(Int, length(tree_queries_list) / 2)
    p_dist = plot(hist_plots..., layout=(rows, 2), size=(1200, 300 * rows))

    dist_plot_path = joinpath(results_dir, "reward_distribution_histograms_$(algo_name).png")
    savefig(p_dist, dist_plot_path)

    @info "Saved statistics CSV: $stats_csv_path"
    @info "Saved summary plot: $summary_plot_path"
    @info "Saved distribution plot: $dist_plot_path"
end

# png(p, "POMCPOW_SubHunt")