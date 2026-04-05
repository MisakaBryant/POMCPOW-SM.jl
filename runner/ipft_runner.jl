using POMDPs
using POMDPModels
using POMDPTools
using SubHunt
using LaserTag
using Random
using Statistics
using Printf
using Plots
using Logging, LoggingExtras
using ParticleFilters

include("../pomdps/LunarLander/po_lunar.jl")
include("../pomdps/WindFarmPOMDP/SensorPlacementPhase/src/SensorPP.jl")
include("../IPFT/src/IPFT.jl")
using .IPFT

logger = TeeLogger(
    global_logger(),
    FileLogger("output.log"; append=true)
)
global_logger(logger)

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

struct RunnerResampler
    n::Int
    lowvar::LowVarianceResampler
end

RunnerResampler(n::Int) = RunnerResampler(n, LowVarianceResampler(n))

function ParticleFilters.resample(r::RunnerResampler, d, rng::AbstractRNG)
    if d isa ParticleCollection || d isa AbstractParticleBelief
        return ParticleFilters.resample(r.lowvar, d, rng)
    end
    return ParticleCollection([rand(rng, d) for _ in 1:r.n])
end

# Return updater-like object required by IBMDPSolver.
# NamedTuple is sufficient because IBMDPSolver only accesses .resample and .max_frac_replaced.
function default_ibmdp_updater(::POMDP; n_particles=300, max_frac_replaced=0.05)
    return (resample=RunnerResampler(n_particles), max_frac_replaced=max_frac_replaced)
end

function trail_ipft(policy::IPFTPlanner; max_steps=100)
    mdp = policy.mdp
    b0 = resample(mdp.resample, initialstate(mdp.pomdp), policy.rng)
    i0 = information(mdp.im, bp=b0)
    s = InformationBelief(b0, i0)

    total_task_reward = 0.0
    for t in 0:(max_steps-1)
        if isterminal(mdp, s)
            break
        end
        a = action(policy, s)
        sp, r = @gen(:sp, :r)(mdp, s, a, policy.rng)
        total_task_reward += r[1] * discount(mdp)^t
        s = sp
    end
    return total_task_reward
end

function solve_and_evaluate_ipft(solver::IBMDPSolver, pomdp::POMDP; max_steps=100, total=1000)
    policy = solve(solver, pomdp)
    rewards = Vector{Float64}(undef, total)
    Threads.@threads for i in 1:total
        local_policy = deepcopy(policy)
        rewards[i] = trail_ipft(local_policy, max_steps=max_steps)
    end
    return summarize_rewards(rewards), rewards
end

# -------- Scenario config (align with runner2 for easier comparison) --------
pomdp = SubHuntPOMDP()
k_o = 6.0
alpha_o = 1/100
tree_queries_list = [5000, 10000, 20000, 50000, 80000]

# pomdp = LightDark1D()
# k_o = 5.0
# alpha_o = 1/15
# tree_queries_list = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

total = 1000
max_depth = 20
max_time = 100.0

# Information objective settings for IBMDPSolver.
ifm = DiscreteEntropy()
discount_information_gain = true

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

stats_table = NamedTuple[]
all_rewards_by_query = Dict{Int, Vector{Float64}}()

@info "Running experiment with algorithm=IPFT_IBMDP"
for tree_queries in tree_queries_list
    rng = MersenneTwister(13)
    ipft = IPFTSolver(depth=max_depth,
                      n_iterations=tree_queries,
                      max_time=max_time,
                      enable_action_pw=false,
                      k_observation=k_o,
                      alpha_observation=alpha_o,
                      rng=rng)
    solver = IBMDPSolver(ipft, default_ibmdp_updater, ifm, discount_information_gain)

    stats, rewards = solve_and_evaluate_ipft(solver, pomdp, total=total)
    @info @sprintf("algorithm=%s, tree_queries=%d, n=%d, min=%.4f, max=%.4f, mean=%.4f, var=%.4f, std=%.4f, se=%.4f",
                   "IPFT_IBMDP", tree_queries, stats.n, stats.min, stats.max, stats.mean, stats.var, stats.std, stats.se)

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

stats_csv_path = joinpath(results_dir, "reward_stats_by_tree_queries_IPFT_IBMDP.csv")
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
                 title="[IPFT_IBMDP] Reward statistics over $(total) trials")
scatter!(p_summary, valid_queries, min_list, marker=:diamond, label="Min")
scatter!(p_summary, valid_queries, max_list, marker=:utriangle, label="Max")

summary_plot_path = joinpath(results_dir, "reward_summary_vs_tree_queries_IPFT_IBMDP.png")
savefig(p_summary, summary_plot_path)

hist_plots = [histogram(all_rewards_by_query[q],
                        bins=30,
                        xlabel="Reward",
                        ylabel="Count",
                        title="[IPFT_IBMDP] tree_queries=$q",
                        legend=false) for q in valid_queries]

rows = ceil(Int, length(valid_queries) / 2)
p_dist = plot(hist_plots..., layout=(rows, 2), size=(1200, 300 * rows))

dist_plot_path = joinpath(results_dir, "reward_distribution_histograms_IPFT_IBMDP.png")
savefig(p_dist, dist_plot_path)

@info "Saved statistics CSV: $stats_csv_path"
@info "Saved summary plot: $summary_plot_path"
@info "Saved distribution plot: $dist_plot_path"
