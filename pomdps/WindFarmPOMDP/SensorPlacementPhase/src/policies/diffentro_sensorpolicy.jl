"""
    DiffEntroPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}

Constructor:

    `DiffEntroPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

mutable struct DiffEntroPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
DiffEntroPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=POMDPPolicies.NothingUpdater()) = DiffEntroPolicy(rng, problem, updater)
DiffEntroPolicy(problem::Union{POMDP,MDP}, extra_params::Vector) = DiffEntroPolicy(problem)


function greedyDiffEntroPolicy(gpla_wf::GPLA, legal_actions::Vector{CartesianIndex{3}}, pomdp::WindFarmPOMDP)
    legal_actions = CartIndices_to_Array([item for item in legal_actions if item[3]==pomdp.altitudes[end]])
    
    conditional_entropy(σ²) = 0.5 * log(2 * pi * exp(1) * σ²)

    ### After taking the new action ###

    function conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        
        # Create temp kernel 
        expanded = CartIndices_to_Array(expand_action_to_below_altitudes(Vector_to_CartIndices(a), pomdp.altitudes))
        complements = hcat(gpla_wf.x, expanded)
        y_hat = zeros(size(complements, 2))
        gpla_temp = GPLA(complements, y_hat, wfparams.num_neighbors, 0, 0, gpla_wf.mean, gpla_wf.kernel, gpla_wf.logNoise.value)

        all_actions = CartIndices_to_Array(POMDPs.actions(pomdp))
        μ, σ² = GaussianProcesses.predict_f(gpla_temp, all_actions)
        return sum(conditional_entropy.(σ²))
    end


    parallel_results = []

    Threads.@threads for idx in 1:size(legal_actions, 2)
        a = legal_actions[:,idx]
        entro = conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        @show (idx,entro)
        push!(parallel_results, (idx,entro))
    end

    conditional_entropy_of_complement_actions = [item[2] for item in sort(parallel_results)]    

    diffentros =  - conditional_entropy_of_complement_actions
    best_vals = argmaxall(diffentros; threshold = 0.0)

    # if more than one best action, choose the one closest to the centroid of legal actions.
    if length(best_vals) == 1
        return legal_actions[:, best_vals]
    else
        best_actions = legal_actions[:, best_vals]
        centroid = StatsBase.mean(legal_actions, dims=2)
        dists_to_centroid = euclidean_dist.(eachcol(best_actions), Ref(centroid))

        closest_action = best_actions[:, argmin(dists_to_centroid)]
        closest_action[end] = wfparams.altitudes[end]
        return closest_action
    end
end

function POMDPs.action(policy::DiffEntroPolicy, b::WindFarmBelief)
    pomdp = policy.problem
    gpla_wf = b.gpla_wf
    legal_actions = actions(pomdp, b)
    policy_action = greedyDiffEntroPolicy(gpla_wf, legal_actions, pomdp)                      # deterministically choose the best location.
    # @show policy_action
    return Vector_to_CartIndices(policy_action)
end
