using DelimitedFiles
using DataFrames
using Statistics
include("../src/utils/parse_functions.jl")

csv_dir = isempty(Main.ARGS) ? "./Generic_Run_Results/" : ARGS[1]

layouttypes = [:greedy,
               :genetic,
               :mcmc
]

parse_results(layouttypes, csv_dir)