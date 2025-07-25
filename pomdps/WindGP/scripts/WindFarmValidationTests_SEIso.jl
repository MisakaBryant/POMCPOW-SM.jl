using Distributions
using LinearAlgebra: diag, isposdef
using Random
using GaussianProcesses
import Statistics
using Optim
using Dates
include("../src/dataparser_GWA.jl")
include("../src/utils/misc.jl")
# include("../src/utils/GP_MCMC_with_tqdm.jl")
include("../src/GPLA.jl")

include("../src/utils/WLK_SEIso.jl")

#=========== PARAMS ==========#
# Parsing Farm Data
farm = "AltamontCA"
grid_dist = 220
altitudes = [100, 150, 200]
nx = 90
ny = nx

# Training Set
SPLIT_NUM = 5   # choose odd number
nx_start = div(nx, SPLIT_NUM) * div(SPLIT_NUM, 2)
nx_end = div(nx, SPLIT_NUM) * (SPLIT_NUM - div(SPLIT_NUM, 2))
ny_start = nx_start
ny_end = nx_end

# Observation Set
grid_dist_obs_vals = grid_dist .* [2, 5, 10]

# GPLA
σy = -1.7
NUM_NEIGHBORS = 5

# Optimization
PSWARM_LOWER = [-2.0, 4.0, 0.0, -5.0, 0.0, -5.0, 0.0, 0.0]
PSWARM_UPPER = [2.0, 10.0, 10.0, 1.0, 5.0, 1.0, 10.0, 0.5]
OPTIM_METHOD = ParticleSwarm(lower=PSWARM_LOWER, upper=PSWARM_UPPER, n_particles=10)


#========= KERNEL PARAMS ========#
# SquaredExponentialKernel PARAMS
l_sq = exp(7)
σs_sq = exp(0.7)

# LinearExponentialKernel PARAMS
l_lin = 10000.0
σs_lin = 1.0

# WindLogLawKernel PARAMS
d = 0.0
zₒ = 0.05
#=================================#


# Load wind farm data
Map = get_3D_data(farm; altitudes=altitudes)

function get_dataset(Map, altitudes, grid_dist_mid, grid_dist, nx_start, nx_end, ny_start, ny_end)
    X_set0 = []
    Y_set0 = Float64[]
    
    for h in altitudes
        append!(X_set0, [[i, j, Float64(h)] for j in Float64(ny_start-1)*grid_dist : grid_dist_mid : Float64(ny_end-1)*grid_dist for i in Float64(nx_start-1)*grid_dist : grid_dist_mid : Float64(nx_end-1)*grid_dist])
        append!(Y_set0, vec(Map[h][nx_start : nx_end, ny_start : ny_end]))
    end
    
    X_set = transform4GPjl(X_set0)
    Y_set = Y_set0  

    return X_set, Y_set
end

function get_Y_from_farm_location(loc, Map, grid_dist, nx, ny, h)
    idx = loc[1:2]./ grid_dist .+ 1.0
    return Map[h][Int.(idx)...]
end


# Create field set
X_field, Y_field = get_dataset(Map, altitudes, grid_dist, grid_dist, 1, nx, 1, ny)


# Create training sets
X_train1, Y_train1 = get_dataset(Map, altitudes, grid_dist, grid_dist, 1, div(nx, SPLIT_NUM), 1, div(ny, SPLIT_NUM))
X_train2, Y_train2 = get_dataset(Map, altitudes, grid_dist, grid_dist, div(nx, SPLIT_NUM) * (SPLIT_NUM - 1) + 1, nx, 1, div(ny, SPLIT_NUM))
X_train3, Y_train3 = get_dataset(Map, altitudes, grid_dist, grid_dist, 1, div(nx, SPLIT_NUM), div(ny, SPLIT_NUM) * (SPLIT_NUM - 1) + 1, ny)
X_train4, Y_train4 = get_dataset(Map, altitudes, grid_dist, grid_dist, div(nx, SPLIT_NUM) * (SPLIT_NUM - 1) + 1, nx, div(ny, SPLIT_NUM) * (SPLIT_NUM - 1) + 1, ny)
X_train5, Y_train5 = get_dataset(Map, altitudes, grid_dist, grid_dist, div(nx, SPLIT_NUM) * div(SPLIT_NUM, 2) + 1, div(nx, SPLIT_NUM) * (SPLIT_NUM - div(SPLIT_NUM, 2)), div(ny, SPLIT_NUM) * div(SPLIT_NUM, 2) + 1, div(ny, SPLIT_NUM) * (SPLIT_NUM - div(SPLIT_NUM, 2)))

TRAIN_DATA = Dict()
push!(TRAIN_DATA, (1 => (X_train1, Y_train1)))
push!(TRAIN_DATA, (2 => (X_train2, Y_train2)))
push!(TRAIN_DATA, (3 => (X_train3, Y_train3)))
push!(TRAIN_DATA, (4 => (X_train4, Y_train4)))
push!(TRAIN_DATA, (5 => (X_train5, Y_train5)))


# Create observation sets
X_obs1, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[1], grid_dist, 1, nx, 1, ny)
Y_obs1 = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs1)))   # Retrieve the Y values on the Map for these coordinates.

X_obs2, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[2], grid_dist, 1, nx, 1, ny)
Y_obs2 = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs2)))   # Retrieve the Y values on the Map for these coordinates.

X_obs3, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[3], grid_dist, 1, nx, 1, ny)
Y_obs3 = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs3)))   # Retrieve the Y values on the Map for these coordinates.

OBS_DATA = Dict()
push!(OBS_DATA, (1 => (X_obs1, Y_obs1)))
push!(OBS_DATA, (2 => (X_obs2, Y_obs2)))
push!(OBS_DATA, (3 => (X_obs3, Y_obs3)))

# Create initial kernel
kern_gp_wll = WLK_SEIso(log(l_sq), log(σs_sq), log(l_lin), log(σs_lin), d, zₒ)


#========= CROSS VALIDATION TESTS ========#

# Intrafield Validations

function LogLikelihood(y_truth, μ, σ)
    Σ = σ^2 + eps()
    constantTerm = -0.5 * log(2*pi)
    modelComplexityTerm = -0.5 * log(Σ)
    dataFitTerm = -0.5 * ((y_truth - μ)^2 * (1/Σ))
    
    return constantTerm + modelComplexityTerm + dataFitTerm
end

CROSS_VAL_INTRA_RESULTS = Dict()

for idx_t in 1:5
    
    println("## Starting Optimization for idx_t=$idx_t")
    X_train, Y_train = TRAIN_DATA[idx_t]
    @time gpla_WF = GPLA(X_train, Y_train, NUM_NEIGHBORS, 0, 0, MeanConst(mean(Y_train)), kern_gp_wll, σy)
    optimize!(gpla_WF, method = OPTIM_METHOD)
    optim_params = GaussianProcesses.get_params(gpla_WF)
    
    for idx_o in 1:3

        println("## Calculating logL for idx_o=$idx_o")
        X_obs, Y_obs = OBS_DATA[idx_o]
        GaussianProcesses.fit!(gpla_WF, X_obs, Y_obs)

        # Calculate average log likelihood
        μ_gpla_WF, Σ_gpla_WF = predict_f(gpla_WF, X_field)
        logL = map(idx -> LogLikelihood(Y_field[idx], μ_gpla_WF[idx], Σ_gpla_WF[idx]), 1:length(Y_field))
        avgLogL = average(logL)
        @show avgLogL
        @show optim_params

        push!(CROSS_VAL_INTRA_RESULTS, ((idx_t=idx_t, idx_o=idx_o) => (avgLogL=avgLogL, optim_params=optim_params, size_obs=length(Y_obs))))

    end
end



# Interfield Validations

# Get best Theta and save to disk.
ky = reduce((x, y) -> CROSS_VAL_INTRA_RESULTS[x].avgLogL ≥ CROSS_VAL_INTRA_RESULTS[y].avgLogL ? x : y, keys(CROSS_VAL_INTRA_RESULTS))
best_Theta = CROSS_VAL_INTRA_RESULTS[ky].optim_params

open("WF_val_results_WLK_SEIso.txt", "a") do io
    writedlm(io, [" "; string(Dates.now()); " "; "BEST_THETA"; best_Theta; " "; "CROSS_VAL_INTRA_RESULTS"])
    for (k,v) in CROSS_VAL_INTRA_RESULTS
        writedlm(io, [(k,v)])
    end
end


# Load other farm, and validate.
other_farms = ["BullHillME", "CantonMtME", "DelawareMtTX", "SmokyHillsKS", "WoodwardMtTX"]
CROSS_VAL_INTER_RESULTS = Dict()

for farm in other_farms

    Map = get_3D_data(farm; altitudes=altitudes)
    nx = ny = 60

    # Create field set
    X_field, Y_field = get_dataset(Map, altitudes, grid_dist, grid_dist, 1, nx, 1, ny)


    # Create observation sets
    X_obs1, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[1], grid_dist, 1, nx, 1, ny)
    Y_obs1 = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs1)))   # Retrieve the Y values on the Map for these coordinates.

    X_obs2, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[2], grid_dist, 1, nx, 1, ny)
    Y_obs2 = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs2)))   # Retrieve the Y values on the Map for these coordinates.

    X_obs3, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[3], grid_dist, 1, nx, 1, ny)
    Y_obs3 = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs3)))   # Retrieve the Y values on the Map for these coordinates.

    OBS_DATA = Dict()
    push!(OBS_DATA, (1 => (X_obs1, Y_obs1)))
    push!(OBS_DATA, (2 => (X_obs2, Y_obs2)))
    push!(OBS_DATA, (3 => (X_obs3, Y_obs3)))

    for idx_o in 1:3

        X_obs, Y_obs = OBS_DATA[idx_o]
        @time gpla_WF = GPLA(X_obs, Y_obs, NUM_NEIGHBORS, 0, 0, MeanConst(mean(Y_obs)), kern_gp_wll, σy)
        GaussianProcesses.set_params!(gpla_WF, best_Theta)

        μ_gpla_WF, Σ_gpla_WF = predict_f(gpla_WF, X_field)
        logL = map(idx -> LogLikelihood(Y_field[idx], μ_gpla_WF[idx], Σ_gpla_WF[idx]), 1:length(Y_field))
        avgLogL = average(logL)

        push!(CROSS_VAL_INTER_RESULTS, ((farm=farm, idx_o=idx_o) => (avgLogL=avgLogL, size_obs=length(Y_obs))))

    end
end

open("WF_val_results_WLK_SEIso.txt", "a") do io
    writedlm(io, [" "; "CROSS_VAL_INTER_RESULTS"])
    for (k,v) in CROSS_VAL_INTER_RESULTS
        writedlm(io, [(k,v)])
    end
end



#========= GENERATIVE TESTS ========#

NUM_SAMPLES = 100
SAMPLES = Dict()

QUANT_CHECK = [0.05, 0.5, 0.95]
QUANTILES = Dict()


for farm in tqdm(vcat(farm, other_farms))

    println("## Sampling from $farm...")
    
    Map = get_3D_data(farm; altitudes=altitudes)
    nx, ny = size(Map[100])

    # Create field set
    X_field, Y_field = get_dataset(Map, altitudes, grid_dist, grid_dist, 1, nx, 1, ny)
    push!(QUANTILES, ((farm=farm, id=0) => (quant_check=QUANT_CHECK, quants=quantile(Y_field, QUANT_CHECK))))

    # Create observation sets
    X_obs, _ = get_dataset(Map, altitudes, grid_dist_obs_vals[2], grid_dist, 1, nx, 1, ny)
    Y_obs = map(lambda -> get_Y_from_farm_location(lambda, Map, grid_dist, nx, ny, 150), collect(eachcol(X_obs)))   # Retrieve the Y values on the Map for these coordinates.

    # Create GP with best_Theta
    gpla_WF = GPLA(X_obs, Y_obs, NUM_NEIGHBORS, 0, 0, MeanConst(mean(Y_obs)), kern_gp_wll, σy)
    GaussianProcesses.set_params!(gpla_WF, best_Theta)

    # Sample from the posterior
    for idx_s in tqdm(1:NUM_SAMPLES)        
        Y_sample = rand(gpla_WF, X_field)
        push!(SAMPLES, ((farm=farm, id=idx_s) => (Y_sample=Y_sample)))
        push!(QUANTILES, ((farm=farm, id=idx_s) => (quant_check=QUANT_CHECK, quants=quantile(Y_sample, QUANT_CHECK))))
    end

end


QUANT_STF_THRESHOLD = 0.5  # [m/s] permitted deviation in windspeed quants
QUANT_STF_THRESHOLD_COUNT = Dict(f => [0.0 for _ in QUANT_CHECK] for f in vcat(farm, other_farms))

for farm in vcat(farm, other_farms)
    for idx_s in 1:NUM_SAMPLES
        ky = (farm=farm, id=idx_s)

        QUANT_STF_THRESHOLD_COUNT[farm] += abs.(QUANTILES[ky].quants - QUANTILES[(farm=farm, id=0)].quants) .≤ QUANT_STF_THRESHOLD
    end
    QUANT_STF_THRESHOLD_COUNT[farm] /= NUM_SAMPLES
end

open("WF_val_results_WLK_SEIso.txt", "a") do io
    writedlm(io, [" "; "QUANTILES"])
    for (k,v) in QUANTILES
        writedlm(io, [(k,v)])
    end
    writedlm(io, [" "; "QUANT_STF_THRESHOLD_COUNT for $QUANT_STF_THRESHOLD"])
    for (k,v) in QUANT_STF_THRESHOLD_COUNT
        writedlm(io, [(k,v)])
    end
end


### PLOT STUFF ###
# using Plots


# p1 = heatmap(reshape(Y0, (nx,ny)))
# Plots.savefig(p1, "p1")

# nx_star = length(0.0:grid_dist_obs:(nx-1)*grid_dist)
# ny_star = length(0.0:grid_dist_obs:(ny-1)*grid_dist)

# # predicted values
# # p2 = heatmap(reshape(μ_exact_WF, (nx_star,ny_star)))
# p3 = heatmap(reshape(μ_gpla_WF, (nx, ny)))
# p4 = heatmap(reshape(Y_sample, (nx, ny)))
# Plots.savefig(p4, "p4")