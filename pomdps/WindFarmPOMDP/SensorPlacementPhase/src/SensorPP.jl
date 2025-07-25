# Modules
# using Plots
using Random
using ArgParse
using POMDPs, POMDPTools
using BasicPOMCP, POMCPOW
using Distributions
using Discretizers
using Parameters
using Dates
using ImageTransformations
using NearestNeighbors
using StatsBase
# using BayesianOptimization
using Memoize
import Noise: add_gauss!

# WindGP repo
include("../../../WindGP/src/WindGP.jl")

# SensorPlacementPhase scripts
include("./beliefstates.jl")
include("./windfarmpomdp.jl")
include("./policies.jl")
include("./common.jl")
include("./utils/plot_functions.jl")
include("./utils/parse_functions.jl")

println("### SensorPP.jl loaded ###")