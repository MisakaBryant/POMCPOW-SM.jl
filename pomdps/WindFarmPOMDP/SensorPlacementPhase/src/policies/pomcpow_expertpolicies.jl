"""
    UCBGreedyPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
an expert policy that is used to select the greediest actions with respect to some Upper Confidence Bounds.

Constructor:

    `UCBGreedyPolicy(problem::Union{POMDP,MDP};
             rng=Random.GLOBAL_RNG,
             updater=POMDPPolicies.NothingUpdater())`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `POMDPPolicies.NothingUpdater` in the above constructor)
"""

mutable struct UCBGreedyPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: Policy
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
UCBGreedyPolicy(problem::POMDP; rng=Random.GLOBAL_RNG, updater=POMDPPolicies.NothingUpdater()) = UCBGreedyPolicy(rng, problem, updater)
UCBGreedyPolicy(pomdp::POMDP, extra_params) = UCBGreedyPolicy(pomdp)

function greedyUCBExpert(gpla_wf::GPLA, legal_actions::Vector{CartesianIndex{3}})
    legal_actions = CartIndices_to_Array(legal_actions)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf, legal_actions)

    σ = sqrt.(σ²)
    N = max(1, length(gpla_wf.y))

    z_value = 1.645   # chosen: 90 percent confidence interval
    UCB = μ + z_value / sqrt(N) * σ
    
    # best_val = argmax(vec(UCB))
    # best_action = legal_actions[:,best_val]

    best_vals = argmaxall(vec(UCB); threshold = 1e-6)
    best_action = legal_actions[:,rand(best_vals)]

    return best_action
end

function POMDPs.action(policy::UCBGreedyPolicy, b::WindFarmBelief)
    gpla_wf = b.gpla_wf
    legal_actions = actions(policy.problem, b)
    policy_action = greedyUCBExpert(gpla_wf, legal_actions)                      # deterministically choose the best location.
    # @show policy_action
    return Vector_to_CartIndices(policy_action)
end



""" MCTSRolloutPolicies supertype, and MCTSRolloutUpdater belief updater definitions """

abstract type MCTSRolloutPolicies <: POMDPs.Policy end
struct EmptyBeliefNode <: BasicPOMCP.BeliefNode end

struct MCTSRolloutUpdater <: POMDPs.Updater
    altitudes::Vector{Number}
    grid_dist::Int
end

function POMDPs.update(bu::MCTSRolloutUpdater, old_b::WindFarmBelief, a::CartesianIndex{3}, obs::Vector{Float64})
    a0 = CartIndices_to_Vector(a)
    a = expand_action_to_below_altitudes(a, bu.altitudes)
    a = CartIndices_to_Array(a)

    x_acts = hcat(old_b.x_acts, a0)

    # gpla_wf = deepcopy(old_b.gpla_wf)
    gpla_wf = old_b.gpla_wf

    x_obs, y_obs = gpla_wf.x, gpla_wf.y
    
    x_obs = hcat(x_obs, a)
    y_obs = vcat(y_obs, obs)
    GaussianProcesses.fit!(gpla_wf, x_obs, y_obs)

    return WindFarmBelief(x_acts, gpla_wf)
end



"""
    UCBRolloutPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
an expert policy that is used to select from a set of actions with respect to UCB of latest wind speed belief.

Constructor:

    `UCBRolloutPolicy(problem::Union{POMDP};
             rng=Random.GLOBAL_RNG,
             updater=MCTSRolloutUpdater(POMDP))`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `MCTSRolloutUpdater` in the above constructor)
"""

mutable struct UCBRolloutPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: MCTSRolloutPolicies
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
UCBRolloutPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=MCTSRolloutUpdater(problem.altitudes, problem.grid_dist)) = UCBRolloutPolicy(rng, problem, updater)


@memoize function get_UCB_rollout_actions(gpla_wf_rollout::GPLA, legal_actions::Vector{CartesianIndex{3}}; top_n_to_consider::Int = 15)

    legal_actions = CartIndices_to_Array(legal_actions)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf_rollout, legal_actions)
    σ = sqrt.(σ²)
    N = length(gpla_wf_rollout.y) + 1    # Laplace smoothing to prevent division by zero.

    z_value = 1.645   # chosen: 90 percent confidence interval
    UCB = μ + z_value / sqrt(N) * σ
    
    best_vals = partialsortperm(vec(UCB), 1:top_n_to_consider, rev=true)

    best_actions = legal_actions[:,best_vals]
    weights = UCB[best_vals]
    return best_actions, weights
end

# function POMDPPolicies.action(policy::UCBRolloutPolicy, b::WindFarmBelief)
#     gpla_wf_rollout = b.gpla_wf
#     legal_actions = actions(policy.problem, b)
#     best_actions, weights = get_UCB_rollout_actions(gpla_wf_rollout, legal_actions)
#     policy_action = StatsBase.sample(collect(eachcol(best_actions)), Weights(weights))        # sample one candidate action w.r.t weights.
#     return Vector_to_CartIndices(policy_action)
# end

function UCBWideningPolicy(pomdp::WindFarmPOMDP, b::WindFarmBelief, h::BeliefNode)
    gpla_wf_rollout = b.gpla_wf
    legal_actions = actions(pomdp, b)
    best_actions, weights = get_UCB_rollout_actions(gpla_wf_rollout, legal_actions)
    policy_action = StatsBase.sample(collect(eachcol(best_actions)), Weights(weights))        # sample one candidate action w.r.t weights.
    return Vector_to_CartIndices(policy_action)
    # return Vector_to_CartIndices(best_actions[:,1])    # debug
end

POMDPs.updater(policy::UCBRolloutPolicy) = policy.updater
POMDPs.action(policy::UCBRolloutPolicy, b::WindFarmBelief) = UCBWideningPolicy(policy.problem, b::WindFarmBelief, EmptyBeliefNode())

function UCBWideningPolicy(pomdp::WindFarmPOMDP, sf::POMCPOW.StateBelief, nd::POWTreeObsNode)
    legal_actions = actions(pomdp, sf.sr_belief.dist.items[end][1])
    return rand(legal_actions)
end 

"""
    MIRolloutPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater}
an expert policy that is used to select from a set of actions with respect to MI of latest wind speed belief.

Constructor:

    `MIRolloutPolicy(problem::Union{POMDP};
             rng=Random.GLOBAL_RNG,
             updater=MCTSRolloutUpdater(POMDP))`

# Fields 
- `rng::RNG` a random number generator 
- `probelm::P` the POMDP or MDP problem 
- `updater::U` a belief updater (default to `MCTSRolloutUpdater` in the above constructor)
"""

mutable struct MIRolloutPolicy{RNG<:AbstractRNG, P<:Union{POMDP,MDP}, U<:Updater} <: MCTSRolloutPolicies
    rng::RNG
    problem::P
    updater::U # set this to use a custom updater, by default it will be a void updater
end
# The constructor below should be used to create the policy so that the action space is initialized correctly
MIRolloutPolicy(problem::Union{POMDP,MDP}; rng=Random.GLOBAL_RNG, updater=MCTSRolloutUpdater(problem.altitudes, problem.grid_dist)) = MIRolloutPolicy(rng, problem, updater)


function get_MI_rollout_actions(gpla_wf_rollout::GPLA, legal_actions::Vector{CartesianIndex{3}}; top_n_to_consider::Int = 100)

    legal_actions = CartIndices_to_Array(legal_actions)

    # Part 1.
    conditional_entropy_of_actions = conditional_entropy_of_action1.(Ref(gpla_wf_rollout), eachcol(legal_actions))

    # Part 2.
    results = []
    Threads.@threads for idx in 1:size(legal_actions, 2)
        a = legal_actions[:,idx]
        entro = conditional_entropy_of_complement_action1(gpla_wf_rollout, pomdp, a)
        @show (idx,entro)
        push!(results, (idx,entro))
    end

    # Combining Part 1 and 2.
    conditional_entropy_of_complement_actions = [item[2] for item in sort(results)]
    
    MutualInfos = conditional_entropy_of_actions - conditional_entropy_of_complement_actions[1:length(conditional_entropy_of_actions)]
    best_vals = partialsortperm(MutualInfos, 1:top_n_to_consider, rev=true)

    best_actions = legal_actions[:,best_vals]
    weights = MutualInfos[best_vals]
    return best_actions, weights
end

# function POMDPPolicies.action(policy::MIRolloutPolicy, b::WindFarmBelief)
#     gpla_wf_rollout = b.gpla_wf
#     legal_actions = actions(policy.problem, b)
#     best_actions, weights = get_MI_rollout_actions(gpla_wf_rollout, legal_actions)
#     policy_action = StatsBase.sample(collect(eachcol(best_actions)), Weights(weights))        # sample one candidate action w.r.t weights.
#     return Vector_to_CartIndices(policy_action)
# end

function MIWideningPolicy(pomdp::WindFarmPOMDP, b::WindFarmBelief, h::BeliefNode)
    gpla_wf_rollout = b.gpla_wf
    legal_actions = actions(pomdp, b)
    best_actions, weights = get_MI_rollout_actions(gpla_wf_rollout, legal_actions)
    policy_action = StatsBase.sample(collect(eachcol(best_actions)), Weights(weights))        # sample one candidate action w.r.t weights.
    return Vector_to_CartIndices(policy_action)
end

POMDPs.updater(policy::MIRolloutPolicy) = policy.updater
POMDPs.action(policy::MIRolloutPolicy, b::WindFarmBelief) = MIWideningPolicy(policy.problem, b::WindFarmBelief, EmptyBeliefNode())