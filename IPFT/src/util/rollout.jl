"""
To use RolloutEstimator(policy) where policy is not just a random policy, but a heuristic policy
"""
function MCTS.convert_estimator(ev::RolloutEstimator, solver::IPFTSolver, mdp::InformationRewardBeliefMDP)
    return MCTS.SolvedRolloutEstimator(MCTS.convert_to_policy(ev.solver, mdp.pomdp),
                                       solver.rng,
                                       solver.depth,
                                       0.0)
end

function MCTS.convert_estimator(est::FOValue, solver::IPFTSolver, mdp::InformationRewardBeliefMDP)
    return MCTS.convert_estimator(est, solver, mdp.pomdp)
end

"""
Rollout estimator for information reward belief MDP
"""
function MCTS.estimate_value(estimator::BasicPOMCP.SolvedFOValue, bmdp::InformationRewardBeliefMDP, ib::InformationBelief{T}, steps::Int) where {T<:ParticleCollection}
    ps = particles(ib.b)
    isempty(ps) && return [0.0, 0.0]

    # SolvedFOValue policies (e.g., ValueIterationPolicy on UnderlyingMDP)
    # are state-based, so approximate belief value by particle mean.
    v = mean(POMDPs.value(estimator.policy, s) for s in ps)
    [v, 0.0]
end
