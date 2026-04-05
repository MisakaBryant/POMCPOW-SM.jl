module IPFT

using Distributions
using Random
using Statistics
using StatsBase: sample, mean, weights
import StatsBase: entropy
using LinearAlgebra

using POMDPs
using POMDPTools
using POMDPSimulators
using MCTS
using MCTS: convert_estimator
using BasicPOMCP
using ParticleFilters
using QMDP

using Test
# using InteractiveUtils

# Compatibility helper: route legacy generate_sr call sites to current @gen API.
generate_sr(mdp, s, a, rng) = @gen(:sp, :r)(mdp, s, a, rng)

# Compatibility helper: CPUTime.jl-free microsecond timer.
CPUtime_us() = time_ns() / 1.0e3


include("information_measures/InformationMeasures.jl")

export
    AbstractMultiObjectiveCriterion,
    MultiObjective,
    MultiObjectiveUCB,
    q_value
include("util/action_selection.jl")

export
    InformationMCTSPlanner,
    InformationBelief,
    pdf,
    resample
include("util/util.jl")

export
    IPFTPlanner,
    IPFTSolver,
    IPFTStateNode,
    IPFTree,
    action_info,
    InformationRewardBeliefMDP
include("core/ipft_types.jl")
include("core/ipft.jl")
include("core/information_mdp.jl")

export
    InformationFilter,
    update,
    initialize_belief
include("util/information_filter.jl")

include("util/rollout.jl")

export
    IBMDPSolver
include("util/ibmdp_solver.jl")

end # module
