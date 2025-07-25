"""
    MutualInfoPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}

Constructor:

    `MutualInfoPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

mutable struct MutualInfoPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
MutualInfoPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=POMDPPolicies.NothingUpdater()) = MutualInfoPolicy(rng, problem, updater)
MutualInfoPolicy(problem::Union{POMDP,MDP}, extra_params::Vector) = MutualInfoPolicy(problem)



function greedyMutualInfoPolicy(gpla_wf::GPLA, legal_actions::Vector{CartesianIndex{3}}, pomdp::WindFarmPOMDP)
    legal_actions = CartIndices_to_Array([item for item in legal_actions if item[3]==pomdp.altitudes[end]])

    # Part 1.
    conditional_entropy_of_actions = conditional_entropy_of_action1.(Ref(gpla_wf), eachcol(legal_actions))

    # Part 2.
    results = []
    Threads.@threads for idx in 1:size(legal_actions, 2)
        a = legal_actions[:,idx]
        entro = conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
        @show (idx,entro)
        push!(results, (idx,entro))
    end

    # Combining Part 1 and 2.
    conditional_entropy_of_complement_actions = [item[2] for item in sort(results)]
    
    MutualInfos = conditional_entropy_of_actions - conditional_entropy_of_complement_actions
    best_vals = argmaxall(MutualInfos; threshold = 0.0)

    # If more than one best action, choose the one closest to the centroid of legal actions.
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

conditional_entropy(σ²) = 0.5 * log(2 * pi * exp(1) * σ²)

function conditional_entropy_of_action1(gpla_wf, a)
        
    μ, σ² = GaussianProcesses.predict_f(gpla_wf, a)
    σ² = σ²[1]    # convert 1×1 Array to Number.
    return conditional_entropy(σ²)
end

function get_complement_actions(gpla_wf, pomdp, a)
        
    # Set{CartesianIndex} of all actions.
    all_actions_Set = Set(POMDPs.actions(pomdp))

    # Set{CartesianIndex} of the previous actions.
    prev_actions_Set = Vector_to_CartIndices.(eachcol(gpla_wf.x))

    # Set{CartesianIndex} of the actions that will be observed if `a` is taken.
    expanded = expand_action_to_below_altitudes(Vector_to_CartIndices(a), pomdp.altitudes)
    
    setdiff!(all_actions_Set, prev_actions_Set)
    setdiff!(all_actions_Set, expanded)
    
    complements = CartIndices_to_Array(collect(all_actions_Set))
    return complements
end

function conditional_entropy_of_complement_action1(gpla_wf, pomdp, a)
    
    # Create temp kernel 
    complements = get_complement_actions(gpla_wf, pomdp, a)
    y_hat = zeros(size(complements, 2))
    gpla_temp = GPLA(complements, y_hat, wfparams.num_neighbors, 0, 0, gpla_wf.mean, gpla_wf.kernel, gpla_wf.logNoise.value)

    μ, σ² = GaussianProcesses.predict_f(gpla_temp, a)
    σ² = σ²[1]    # convert 1×1 Array to Number.
    return conditional_entropy(σ²)
end

function POMDPs.action(policy::MutualInfoPolicy, b::WindFarmBelief)
    pomdp = policy.problem
    gpla_wf = b.gpla_wf
    legal_actions = actions(pomdp, b)
    policy_action = greedyMutualInfoPolicy(gpla_wf, legal_actions, pomdp)                      # deterministically choose the best location.
    # @show policy_action
    return Vector_to_CartIndices(policy_action)
end
