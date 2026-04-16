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
using DiscreteValueIteration

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

function summarize_trial_terminations(ib_terminal_flags::AbstractVector{Bool},
                                      ib_only_terminal_flags::AbstractVector{Bool},
                                      latent_terminal_flags::AbstractVector{Bool},
                                      max_steps_flags::AbstractVector{Bool})
    n = length(ib_terminal_flags)
    ib_terminal_count = count(identity, ib_terminal_flags)
    ib_only_terminal_count = count(identity, ib_only_terminal_flags)
    latent_terminal_count = count(identity, latent_terminal_flags)
    max_steps_count = count(identity, max_steps_flags)
    return (
        n=n,
        ib_terminal_count=ib_terminal_count,
        ib_terminal_rate=n > 0 ? ib_terminal_count / n : 0.0,
        ib_only_terminal_count=ib_only_terminal_count,
        ib_only_terminal_rate=n > 0 ? ib_only_terminal_count / n : 0.0,
        latent_terminal_count=latent_terminal_count,
        latent_terminal_rate=n > 0 ? latent_terminal_count / n : 0.0,
        max_steps_count=max_steps_count,
        max_steps_rate=n > 0 ? max_steps_count / n : 0.0
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

function trail_ipft(policy::IPFTPlanner, pomdp::POMDP; max_steps=100, n_eval_particles=300)
    # Evaluate on the original POMDP trajectory (single latent state),
    # while keeping an information belief only for action selection.
    mdp = policy.mdp
    # Match stepthrough-style randomness: use task-local default RNG directly.
    sim_rng = Random.default_rng()
    up = BootstrapFilter(pomdp, n_eval_particles, sim_rng)
    b0 = initialize_belief(up, initialstate(pomdp))
    ib = InformationBelief(b0, information(mdp.im, bp=b0))
    s = rand(sim_rng, initialstate(pomdp))

    total_task_reward = 0.0
    for t in 0:(max_steps-1)
        latent_terminal = isterminal(pomdp, s)
        ib_terminal = isterminal(mdp, ib)
        if latent_terminal || ib_terminal
            return (
                reward=total_task_reward,
                ib_terminal=ib_terminal,
                ib_only_terminal=ib_terminal && !latent_terminal,
                latent_terminal=latent_terminal,
                max_steps_reached=false
            )
        end

        a = action(policy, ib)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, sim_rng)
        total_task_reward += r * discount(pomdp)^t

        bp = update(up, ib.b, a, o)
        ib = InformationBelief(bp, information(mdp.im, bp=bp))
        s = sp
    end
    # @show total_task_reward
    return (
        reward=total_task_reward,
        ib_terminal=false,
        ib_only_terminal=false,
        latent_terminal=false,
        max_steps_reached=true
    )
end

function solve_and_evaluate_ipft(solver::IBMDPSolver, pomdp::POMDP; max_steps=100, total=1000)
    policy = solve(solver, pomdp)
    rewards = Vector{Float64}(undef, total)
    ib_terminal_flags = falses(total)
    ib_only_terminal_flags = falses(total)
    latent_terminal_flags = falses(total)
    max_steps_flags = falses(total)
    Threads.@threads for i in 1:total
        local_policy = deepcopy(policy)
        # Random.seed!(local_policy, trial_seeds[i])
        outcome = trail_ipft(local_policy, pomdp, max_steps=max_steps)
        rewards[i] = outcome.reward
        ib_terminal_flags[i] = outcome.ib_terminal
        ib_only_terminal_flags[i] = outcome.ib_only_terminal
        latent_terminal_flags[i] = outcome.latent_terminal
        max_steps_flags[i] = outcome.max_steps_reached
    end
    terminal_stats = summarize_trial_terminations(ib_terminal_flags,
                                                  ib_only_terminal_flags,
                                                  latent_terminal_flags,
                                                  max_steps_flags)
    return summarize_rewards(rewards), rewards, terminal_stats
end

# -------- Scenario config (align with runner2 for easier comparison) --------
# pomdp = SubHuntPOMDP()
# c = 17
# k_o = 6.0
# alpha_o = 1/100
# tree_queries_list = [50, 100, 200, 500, 1000]

# vs = ValueIterationSolver()
# if !isdefined(Main, :vp) || vp.mdp != pomdp
#     mdp = UnderlyingMDP(pomdp)
#     vp = solve(vs, mdp)
# end

pomdp = LightDark1D()
c = 90
k_o = 5.0
alpha_o = 1/15
tree_queries_list = [1000, 2000, 5000, 10000, 20000, 50000]

# pomdp = gen_lasertag(rng=MersenneTwister(7), robot_position_known=true)
# c = 26 
# k_o = 4.0
# alpha_o = 1/35
# similarity_threshold = 0.99
# tree_queries_list = [5000, 10000, 20000, 50000]

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
                      criterion=MultiObjectiveUCB([1.0,60.0], c),
                      enable_action_pw=false,
                      k_observation=k_o,
                      alpha_observation=alpha_o,
                    #   estimate_value=FOValue(vp),
                      rng=rng)
    solver = IBMDPSolver(ipft, default_ibmdp_updater, ifm, discount_information_gain)

    stats, rewards, terminal_stats = solve_and_evaluate_ipft(solver, pomdp, total=total)
    @info @sprintf("algorithm=%s, tree_queries=%d, n=%d, min=%.4f, max=%.4f, mean=%.4f, var=%.4f, std=%.4f, se=%.4f",
                   "IPFT_IBMDP", tree_queries, stats.n, stats.min, stats.max, stats.mean, stats.var, stats.std, stats.se)
    @info @sprintf("termination_stats tree_queries=%d, ib_terminal=%d(%.2f%%), ib_only_terminal=%d(%.2f%%), latent_terminal=%d(%.2f%%), max_steps=%d(%.2f%%)",
                   tree_queries,
                   terminal_stats.ib_terminal_count, 100 * terminal_stats.ib_terminal_rate,
                   terminal_stats.ib_only_terminal_count, 100 * terminal_stats.ib_only_terminal_rate,
                   terminal_stats.latent_terminal_count, 100 * terminal_stats.latent_terminal_rate,
                   terminal_stats.max_steps_count, 100 * terminal_stats.max_steps_rate)

    push!(stats_table, (
        tree_queries=tree_queries,
        n=stats.n,
        min=stats.min,
        max=stats.max,
        mean=stats.mean,
        var=stats.var,
        std=stats.std,
        se=stats.se,
        ib_terminal_count=terminal_stats.ib_terminal_count,
        ib_terminal_rate=terminal_stats.ib_terminal_rate,
        ib_only_terminal_count=terminal_stats.ib_only_terminal_count,
        ib_only_terminal_rate=terminal_stats.ib_only_terminal_rate
    ))
    all_rewards_by_query[tree_queries] = rewards
end

stats_csv_path = joinpath(results_dir, "reward_stats_by_tree_queries_IPFT_IBMDP.csv")
open(stats_csv_path, "w") do io
    write(io, "tree_queries,n,min,max,mean,var,std,se,ib_terminal_count,ib_terminal_rate,ib_only_terminal_count,ib_only_terminal_rate\n")
    for row in stats_table
        write(io, @sprintf("%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f,%.10f,%d,%.10f,%d,%.10f\n",
                           row.tree_queries, row.n, row.min, row.max, row.mean, row.var, row.std, row.se,
                           row.ib_terminal_count, row.ib_terminal_rate,
                           row.ib_only_terminal_count, row.ib_only_terminal_rate))
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
