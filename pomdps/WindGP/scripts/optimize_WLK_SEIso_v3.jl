include("../src/WindGP.jl")
using Optim
using ImageTransformations

#=========== PARAMS ==========#

# Parsing Farm Data
farm = "AltamontCA"
grid_dist = 220
altitudes = [100, 150, 200]
nx = 20
ny = nx

# Observation Set
grid_dist_obs = grid_dist #.* 10

# GPLA
NUM_NEIGHBORS = 5
SCALE_FACTOR = 20    # for Mean Lookup

# Optimization
#              ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin 
PSWARM_LOWER = [-10.0, -5.0, -10.0, -5.0]
PSWARM_UPPER = [ 10.0,  5.0,  10.0,  5.0]
OPTIM_METHOD = Optim.ParticleSwarm(lower=PSWARM_LOWER, upper=PSWARM_UPPER, n_particles=100)


#========= KERNEL INIT PARAMS ========#
# SquaredExponentialKernel PARAMS
ℓ2_sq = 7.0
σ2_sq = 0.1

# LinearExponentialKernel PARAMS
ℓ_lin = 0.3
σ2_lin = 0.5

# Noise in GPLA
σy = -1.5


# WindLogLawKernel PARAMS (fixed)
d = 0.0
zₒ = 0.05
#======================================#

function get_init_GPLA()
    # Load prior points for belief GP
    X_obs = reshape(Float64[],3,0)
    Y_obs = Float64[]

    # Load wind farm data
    Map = get_3D_data(farm; altitudes = altitudes, loc_parent = "../../WindGP/data/GWA")
    # X_obs, Y_obs = get_dataset(Map, altitudes, grid_dist_obs, grid_dist, 1, nx, 1, ny, 1)

    # Downsample farm data
    IMG_SIZE = (div(nx, isqrt(SCALE_FACTOR)), div(ny, isqrt(SCALE_FACTOR)))

    Y_mean = Float64[]
    X_mean = reshape(Float64[],3,0)

    for h in altitudes
        img = Map[h][1:nx,1:ny]
        img_ds = ImageTransformations.imresize(img, IMG_SIZE)
        
        img_locs = [[i,j] for i in 0.0:grid_dist:(nx-1)*grid_dist, j in 0.0:grid_dist:(ny-1)*grid_dist]
        img_locs_ds = ImageTransformations.imresize(img_locs, IMG_SIZE)
        X = [[item...,h] for item in vec(img_locs_ds)]
        
        Y_mean = vcat(Y_mean, vec(img_ds))
        X_mean = hcat(X_mean, transform4GPjl(X))
    end

    # Create the lookup mean to the GP, and the WLK_SEIso kernel
    gpla_wf_kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    # gpla_wf_mean = MeanLookup(X_mean, Y_mean)
    gpla_wf_mean = MeanConst(0.0)

    # Create initial kernel
    gpla_wf = GPLA(X_mean, Y_mean, NUM_NEIGHBORS, 0, 0, gpla_wf_mean, gpla_wf_kernel, σy)
    GaussianProcesses.set_params!(gpla_wf, [ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin, d, zₒ]; noise=false, domean=false)

    return gpla_wf
end

function objFunctionValue(gpla_wf, lambda)
    """ Notice that this we will try to minimize the output of this function, therefore the negative of mll is returned. """
    
    ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin = lambda
    GaussianProcesses.set_params!(gpla_wf, [ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin, d, zₒ]; noise=false, domean=false)
    return @show -gpla_wf.mll + norm(lambda)
end

function fetchOptimizedGP(opt_final)
    gpla_wf = get_init_GPLA()
    ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin = opt_final
    GaussianProcesses.set_params!(gpla_wf, [ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin, d, zₒ]; noise=false, domean=false)
    return gpla_wf
end

function plot_gpla(gpla_wf; h=150)
    Map = get_3D_data(farm; altitudes = altitudes, loc_parent = "../../WindGP/data/GWA")
    X_field, Y_field = get_dataset(Map, [h], grid_dist_obs, grid_dist, 1, nx, 1, ny)
    μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
    σ = sqrt.(σ²)

    p2 = Plots.heatmap(reshape(σ, (nx,ny)), title="Wind Farm Initial Belief Variance, h = $(h)m")
    p3 = Plots.heatmap(reshape(μ, (nx,ny)), title="Wind Farm Initial Belief Mean, h = $(h)m")
end


# Optim.jl PARAMS
gpla_wf = get_init_GPLA()
opt_method = OPTIM_METHOD
opt_settings = Optim.Options(show_trace=true, iterations = 300)

opt_init = [ℓ2_sq, σ2_sq, ℓ_lin, σ2_lin]
opt_init0 = deepcopy(opt_init)

result = Optim.optimize(lambda -> objFunctionValue(gpla_wf, lambda), opt_init, opt_method, opt_settings)
@show opt_final = result.minimizer

gp_final = fetchOptimizedGP(opt_final)