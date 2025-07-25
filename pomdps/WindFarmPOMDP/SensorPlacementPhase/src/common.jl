"""
    Common functions used in the Sensor Placement Phase.
"""

function extract_solver_method(pomdp, solvermethod, extra_params...)
    policy_dict = Dict(
        :entropy        => ShannonEntropyPolicy,
        :mutualinfo     => MutualInfoPolicy,
        :diffentro      => DiffEntroPolicy,
        :greedy         => UCBGreedyPolicy,
        :grdynonseq     => GrdyNonSeqPlanner,
        :random         => RandomPlanner,
        :bayesian       => BayesianPlanner,
        :genetic        => GeneticPlanner,
        :pomcpow        => POMCPOWPlanner
    )

    return policy_dict[solvermethod](pomdp, collect(extra_params))
end

extract_solver_method(pomdp, solvermethod) = extract_solver_method(pomdp, solvermethod, [])


function retrieve_solution_from_solver(solvermethod, tlparams, wfparams, pomdp, solver, up, b0, s0, no_of_sensors)
    is_tree_method(x) = x ∈ [:pomcpow, :greedy, :entropy, :mutualinfo, :diffentro]
    need_multiple_threads(x) = x ∈ [:mutualinfo, :diffentro]

    if need_multiple_threads(solvermethod) && Threads.nthreads() == 1
        @warn "ABORTED.\nYou are not running Julia in multiple threads.\nRun e.g. `export JULIA_NUM_THREADS=16` in Terminal before running this script."
        exit()
    end

    if is_tree_method(solvermethod)
        println("### Starting Stepthrough ###")
        global states_history = []
        global belief_history = []
        global actions_history = []
        global obs_history = []
        global rewards_history = []
        for (s, a, r, o, b, t, sp, bp) in stepthrough(pomdp, solver, up, b0, s0, "s,a,r,o,b,t,sp,bp", max_steps=no_of_sensors)
            # @show s
            @show a
            @show o
            @show r
            @show t
            push!(states_history,   sp)
            push!(belief_history,   bp)
            push!(actions_history,  a)
            push!(obs_history,      o)
            push!(rewards_history,  r)
        end
        
        # Show utility of solution
        @show RR = get_ground_truth_profit(states_history, tlparams, wfparams)  # goes to L254 in common.jl

    else
        global soln = get_solution(s0, b0, pomdp, tlparams, wfparams, solver)
        @show RR = get_ground_truth_profit(s0, soln, tlparams, wfparams)
    end

    return RR
end