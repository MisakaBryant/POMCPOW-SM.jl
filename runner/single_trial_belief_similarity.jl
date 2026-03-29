using Random
using Statistics
using Printf
using POMDPs
using BasicPOMCP
using POMDPModels
using POMDPTools
using SubHunt
using Base.Threads

const RESULT_DIR = joinpath(@__DIR__, "results")

using POMCPOW

function trail(policy::Policy, pomdp::POMDP; max_steps=100)
    total_reward = 0.0
    for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, policy, "s,a,r,o,b,t,sp,bp", max_steps=max_steps)
        total_reward += r * discount(pomdp)^t
        if isterminal(pomdp, sp)
            break
        end
    end
    return total_reward
end

pomdp = SubHuntPOMDP()
c = 17
k_o = 6
alpha_o = 1/100
similarity_threshold = 0.99
const SCENARIO_NAME = "SubHunt"

const BIN_EDGES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]

function build_output_paths(scenario_name::String, tree_queries::Int)
    suffix = string(scenario_name, "_q", tree_queries)
    mkpath(RESULT_DIR)
    return Dict(
        "sim_log_prefix" => joinpath(RESULT_DIR, string("belief_similarity_trial_", suffix)),
        "trial_stats" => joinpath(RESULT_DIR, string("belief_similarity_trial_stats_", suffix, ".csv")),
        "overall_summary" => joinpath(RESULT_DIR, string("belief_similarity_overall_summary_", suffix, ".txt")),
    )
end

function parse_similarity_log(path::String)
    similarities = Float64[]
    particles_b1 = Int[]
    particles_b2 = Int[]

    if !isfile(path)
        return similarities, particles_b1, particles_b2
    end

    lines = readlines(path)
    for line in lines[2:end]
        fields = split(line, ',')
        if length(fields) != 3
            continue
        end
        push!(similarities, parse(Float64, fields[1]))
        push!(particles_b1, parse(Int, fields[2]))
        push!(particles_b2, parse(Int, fields[3]))
    end

    return similarities, particles_b1, particles_b2
end

function histogram_counts(values::Vector{Float64}, edges::Vector{Float64})
    counts = zeros(Int, length(edges) - 1)
    for x in values
        for i in 1:length(counts)
            left = edges[i]
            right = edges[i + 1]
            if i < length(counts)
                if left <= x < right
                    counts[i] += 1
                    break
                end
            else
                if left <= x <= right
                    counts[i] += 1
                    break
                end
            end
        end
    end
    return counts
end

function safe_stats(values::Vector{T}) where {T<:Real}
    if isempty(values)
        return Dict(
            "count" => 0,
            "min" => NaN,
            "max" => NaN,
            "mean" => NaN,
            "std" => NaN,
            "q25" => NaN,
            "q50" => NaN,
            "q75" => NaN,
        )
    end
    fv = Float64.(values)
    return Dict(
        "count" => length(values),
        "min" => minimum(fv),
        "max" => maximum(fv),
        "mean" => mean(fv),
        "std" => std(fv),
        "q25" => quantile(fv, 0.25),
        "q50" => quantile(fv, 0.5),
        "q75" => quantile(fv, 0.75),
    )
end

function run_single_trial(seed::Int, sim_log_path::String; tree_queries::Int=2000)
    rng = MersenneTwister(seed)

    solver = POMCPOWSolver(
        tree_queries=tree_queries,
        criterion=MaxUCB(c),
        final_criterion=MaxTries(),
        max_depth=20,
        max_time=100,
        enable_action_pw=true,
        k_observation=k_o,
        alpha_observation=alpha_o,
        check_repeat_obs=true,
        rng=rng,
        similarity_threshold=similarity_threshold,
    )

    ENV["POMCPOW_BELIEF_SIM_LOG"] = sim_log_path

    isfile(sim_log_path) && rm(sim_log_path)
    open(sim_log_path, "w") do io
        println(io, "similarity,n_particles_b1,n_particles_b2")
    end

    policy = solve(solver, pomdp)
    total_reward = trail(policy, pomdp, max_steps=100)
    similarities, particles_b1, particles_b2 = parse_similarity_log(sim_log_path)

    sim_stats = safe_stats(similarities)
    p1_stats = safe_stats(particles_b1)
    p2_stats = safe_stats(particles_b2)
    hist_counts = histogram_counts(similarities, BIN_EDGES)

    return Dict(
        "seed" => seed,
        "reward" => total_reward,
        "n_similarity" => length(similarities),
        "sim_stats" => sim_stats,
        "p1_stats" => p1_stats,
        "p2_stats" => p2_stats,
        "hist_counts" => hist_counts,
    ), similarities, particles_b1, particles_b2
end

function run_trials(output_paths::Dict{String,String}; n_trials::Int=5, base_seed::Int=13, tree_queries::Int=2000)
    sim_log_prefix = output_paths["sim_log_prefix"]
    trial_stats_csv = output_paths["trial_stats"]
    overall_summary_txt = output_paths["overall_summary"]

    trial_results = Vector{Dict}(undef, n_trials)
    sims_by_trial = Vector{Vector{Float64}}(undef, n_trials)
    p1_by_trial = Vector{Vector{Int}}(undef, n_trials)
    p2_by_trial = Vector{Vector{Int}}(undef, n_trials)
    sim_log_paths = Vector{String}(undef, n_trials)

    @threads for trial in 1:n_trials
        seed = base_seed + trial - 1
        sim_log_path = string(sim_log_prefix, "_trial", trial, ".csv")
        task_local_storage(:pomcpow_belief_log_path, sim_log_path)
        result, sims, p1, p2 = run_single_trial(seed, sim_log_path, tree_queries=tree_queries)

        trial_results[trial] = result
        sims_by_trial[trial] = sims
        p1_by_trial[trial] = p1
        p2_by_trial[trial] = p2
        sim_log_paths[trial] = sim_log_path

        sim_stats = result["sim_stats"]
        println("Trial ", trial, " finished. seed=", seed,
                ", reward=", round(result["reward"], digits=4),
                ", n_similarity=", result["n_similarity"],
                ", sim_mean=", round(sim_stats["mean"], digits=4),
                ", thread=", threadid())
    end

    all_similarities = Float64[]
    all_p1 = Int[]
    all_p2 = Int[]
    for trial in 1:n_trials
        append!(all_similarities, sims_by_trial[trial])
        append!(all_p1, p1_by_trial[trial])
        append!(all_p2, p2_by_trial[trial])
    end

    open(trial_stats_csv, "w") do io
        println(io, "trial,seed,reward,n_similarity,sim_min,sim_max,sim_mean,sim_std,sim_q25,sim_q50,sim_q75,p1_mean,p2_mean")
        for trial in 1:n_trials
            seed = base_seed + trial - 1
            result = trial_results[trial]
        sim_stats = result["sim_stats"]
        p1_stats = result["p1_stats"]
        p2_stats = result["p2_stats"]
            @printf(io, "%d,%d,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                trial,
                seed,
                result["reward"],
                result["n_similarity"],
                sim_stats["min"],
                sim_stats["max"],
                sim_stats["mean"],
                sim_stats["std"],
                sim_stats["q25"],
                sim_stats["q50"],
                sim_stats["q75"],
                p1_stats["mean"],
                p2_stats["mean"],
            )
        end
    end

    overall_sim_stats = safe_stats(all_similarities)
    overall_p1_stats = safe_stats(all_p1)
    overall_p2_stats = safe_stats(all_p2)
    overall_hist = histogram_counts(all_similarities, BIN_EDGES)

    open(overall_summary_txt, "w") do io
        println(io, "Overall similarity summary")
        println(io, "n_trials=", n_trials)
        println(io, "tree_queries=", tree_queries)
        println(io, "total_similarity_samples=", overall_sim_stats["count"])
        println(io, "similarity_min=", overall_sim_stats["min"])
        println(io, "similarity_max=", overall_sim_stats["max"])
        println(io, "similarity_mean=", overall_sim_stats["mean"])
        println(io, "similarity_std=", overall_sim_stats["std"])
        println(io, "similarity_q25=", overall_sim_stats["q25"])
        println(io, "similarity_q50=", overall_sim_stats["q50"])
        println(io, "similarity_q75=", overall_sim_stats["q75"])
        println(io, "particle_b1_mean=", overall_p1_stats["mean"])
        println(io, "particle_b1_min=", overall_p1_stats["min"])
        println(io, "particle_b1_max=", overall_p1_stats["max"])
        println(io, "particle_b2_mean=", overall_p2_stats["mean"])
        println(io, "particle_b2_min=", overall_p2_stats["min"])
        println(io, "particle_b2_max=", overall_p2_stats["max"])
        println(io, "histogram_bins=", join(BIN_EDGES, ","))
        println(io, "histogram_counts=", join(overall_hist, ","))
    end

    println("All trials finished.")
    println("Threads used: ", nthreads())
    println("Trial stats CSV: ", trial_stats_csv)
    println("Overall summary: ", overall_summary_txt)
    println("Belief similarity logs: ", join(sim_log_paths, ", "))

    return trial_results, overall_sim_stats, overall_hist, overall_p1_stats, overall_p2_stats
end

function run_query_suite(; query_values::Vector{Int}, n_trials::Int=5, base_seed::Int=13)
    println("Scenario: ", SCENARIO_NAME)
    println("Result dir: ", RESULT_DIR)
    println("Queries: ", join(query_values, ", "))

    for q in query_values
        println("\n=== Running query: ", q, " ===")
        paths = build_output_paths(SCENARIO_NAME, q)
        run_trials(paths, n_trials=n_trials, base_seed=base_seed, tree_queries=q)
    end

    println("\nAll query suites finished.")
end

run_query_suite(query_values=[10000, 20000, 50000, 80000, 100000], n_trials=5, base_seed=13)
