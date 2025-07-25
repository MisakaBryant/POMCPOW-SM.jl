"""
    ShannonEntropyPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}

Constructor:

    `ShannonEntropyPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

mutable struct ShannonEntropyPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
ShannonEntropyPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=POMDPPolicies.NothingUpdater()) = ShannonEntropyPolicy(rng, problem, updater)
ShannonEntropyPolicy(problem::Union{POMDP,MDP}, extra_params::Vector) = ShannonEntropyPolicy(problem)


function greedyShannonPolicy(gpla_wf::GPLA, legal_actions::Vector{CartesianIndex{3}})
    legal_actions = CartIndices_to_Array(legal_actions)

    conditional_entropy(σ²) = 0.5 * log(2 * pi * exp(1) * σ²)

    function conditional_entropy_of_action1(gpla_wf, a)
        
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, a)
        σ² = σ²[1]    # convert 1×1 Array to Number.
        return conditional_entropy(σ²)
    end
    

    shannon_entropies = conditional_entropy_of_action1.(Ref(gpla_wf), eachcol(legal_actions))
    best_vals = argmaxall(shannon_entropies; threshold = 0.0)

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

function POMDPs.action(policy::ShannonEntropyPolicy, b::WindFarmBelief)
    gpla_wf = b.gpla_wf
    legal_actions = actions(policy.problem, b)
    policy_action = greedyShannonPolicy(gpla_wf, legal_actions)                      # deterministically choose the best location.
    # @show policy_action
    return Vector_to_CartIndices(policy_action)
end
