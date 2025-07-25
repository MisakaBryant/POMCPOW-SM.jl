using DelimitedFiles
using DataFrames
using Statistics
using ArgParse
include("../src/utils/parse_functions.jl")

csv_dir = isempty(Main.ARGS) ? "./Generic_Run_Results/" : ARGS[1]

solvermethods = [:pomcpow,
                 :genetic,
                 :bayesian,
                 :greedy,
                 :grdynonseq,
                 :random,
                 :entropy,
                 :mutualinfo,
                 :diffentro
]

parse_results(solvermethods, csv_dir)