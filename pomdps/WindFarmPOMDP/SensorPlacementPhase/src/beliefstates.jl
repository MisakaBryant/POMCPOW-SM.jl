struct WindFarmState
    x_acts::Matrix{Float64}        # actions taken so far.
    x_obs::Matrix{Float64}         # locations observed so far.
    y_obs::Vector{Float64}         # values of locations observed so far.
    x_obs_full::Matrix{Float64}    # locations on entire map, if applicable. only for the actual states, not belief hallucinations during tree search.
    y_obs_full::Vector{Float64}    # values of locations on entire map, if applicable. only for the actual states, not belief hallucinations during tree search.
end

struct WindFarmBelief
    x_acts::Matrix{Float64}
    gpla_wf::GPLA
end

struct WindFarmBeliefUpdater <: POMDPs.Updater
    altitudes::Vector{Number}
    grid_dist::Int
end

function POMDPs.update(bu::WindFarmBeliefUpdater, old_b::WindFarmBelief, a::CartesianIndex{3}, obs::Vector)
    a0 = CartIndices_to_Vector(a)
    a = expand_action_to_below_altitudes(a, bu.altitudes)
    a = CartIndices_to_Array(a)

    x_acts = hcat(old_b.x_acts, a0)

    gpla_wf = deepcopy(old_b.gpla_wf)
    x_obs, y_obs = gpla_wf.x, gpla_wf.y
    
    x_obs = hcat(x_obs, a)
    y_obs = vcat(y_obs, obs)
    GaussianProcesses.fit!(gpla_wf, x_obs, y_obs)

    # println("Belief Updated!")
    return WindFarmBelief(x_acts, gpla_wf)
end

@with_kw struct WindFieldBeliefParams
    # Parsing Farm Data
    farm = "AltamontCA"
    grid_dist = 220
    altitudes = [100, 150, 200]
    nx = 20
    ny = 20
    scale_factor = 20

    # Observation Set
    noise_seed = 123
    grid_dist_obs = grid_dist .* 10

    # GPLA
    num_neighbors = 5

    theta = [-1.5,                          # measurement noise, σy
             4.0,                           # mean, only used when kernel.mean is MeanConst.
             6.952381297661223,             # ℓ2_sq
             0.586877700348630293,          # σ2_sq
             6.106476739802775,             # ℓ_lin
             0.506871639714265196,          # σ2_lin
             0.0,                           # d
             0.05                           # zₒ
    ]

end

function initialize_belief_rollout(s::WindFarmState)

    # Load prior points for belief GP
    X_obs, Y_obs = s.x_obs, s.y_obs

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_perfect(wfparams::WindFieldBeliefParams; is_sparse=false)

    if is_sparse    # initial belief knowledge has no noise, but is sparse.
        grid_dist_obs = wfparams.grid_dist_obs
    else            # initial belief knowledge has no noise, and is the entire field.
        grid_dist_obs = wfparams.grid_dist
    end

    # Load prior points for belief GP
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    X_obs, _ = get_dataset(Map, wfparams.altitudes, grid_dist_obs, wfparams.grid_dist, 1, wfparams.nx+1, 1, wfparams.ny+1)
    Y_obs = map(lambda -> get_Y_from_farm_location(lambda, Map, wfparams.grid_dist), collect(eachcol(X_obs)))

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_noisy(wfparams::WindFieldBeliefParams, windNoise::Number)

    # Load prior points for belief GP
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    X_obs, _ = get_dataset(Map, wfparams.altitudes, wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)

    # Additional noise is added to prior observations
    noise = Distributions.Uniform(-windNoise, +windNoise)
    Y_obs = map(lambda -> get_Y_from_farm_location(lambda, Map, wfparams.grid_dist) + rand(noise), collect(eachcol(X_obs)))

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    gpla_wf.logNoise.value = log(windNoise)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_no_prior(wfparams::WindFieldBeliefParams)

    # Load prior points for belief GP
    X_obs = reshape(Float64[],3,0)
    Y_obs = Float64[]

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, MeanConst(wfparams.theta[2]), kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta)

    # Customize constant mean value
    gpla_wf.mean = MeanConst(10.0)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_belief_lookup(wfparams::WindFieldBeliefParams; has_noise=true)
    """ This version has no observation points in the Gaussian prior, but instead, we the mean function of the GP is 
        a lookup table of the downsampled version of the GWA data of the specified altitudes.
    """
    nx, ny = wfparams.nx, wfparams.ny
    grid_dist = wfparams.grid_dist
    scale_factor = wfparams.scale_factor

    # Load prior points for belief GP
    X_obs = reshape(Float64[],3,0)
    Y_obs = Float64[]

    # Load wind farm data
    Map = get_3D_data(wfparams.farm; altitudes = wfparams.altitudes)
    
    # Downsample farm data
    IMG_SIZE = (div(nx, isqrt(scale_factor)), div(ny, isqrt(scale_factor)))
    
    Y_mean = Float64[]
    X_mean = reshape(Float64[],3,0)
    
    for h in wfparams.altitudes
        img = Map[h][1:nx,1:ny]
        img_ds = ImageTransformations.imresize(img, IMG_SIZE)
        
        img_locs = [[i,j] for i in 0.0:grid_dist:(nx-1)*grid_dist, j in 0.0:grid_dist:(ny-1)*grid_dist]
        img_locs_ds = ImageTransformations.imresize(img_locs, IMG_SIZE)
        X = [[item...,h] for item in vec(img_locs_ds)]
        
        Y_mean = vcat(Y_mean, vec(img_ds))
        X_mean = hcat(X_mean, transform4GPjl(X))
    end

    # Add noise to the values of the Mean function
    Random.seed!(wfparams.noise_seed)
    if has_noise
        add_gauss!(Y_mean, 3.0)
        clamp!(Y_mean, 0, Inf)
    end
    
    # Create the lookup mean to the GP
    gpla_wf_mean = MeanLookup(X_mean, Y_mean, CartIndices_to_Array(actions(pomdp)))

    # Create initial kernel
    kernel = WLK_SEIso(eps(), eps(), eps(), eps(), eps(), eps())
    gpla_wf = GPLA(X_obs, Y_obs, wfparams.num_neighbors, 0, 0, gpla_wf_mean, kernel, wfparams.theta[1])
    GaussianProcesses.set_params!(gpla_wf, wfparams.theta[3:end]; noise=false, domean=false)

    x_acts = reshape(Float64[],3,0)
    return WindFarmBelief(x_acts, gpla_wf)
end

function initialize_state(b0::WindFarmBelief, wfparams::WindFieldBeliefParams)
    """ Called only once when the initial state is created. Assumes full knowledge of field. Should only be used with tree search methods. """
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    x_acts = reshape(Float64[],3,0)
    x_obs = b0.gpla_wf.x
    y_obs = b0.gpla_wf.y  
    X_field, Y_field = get_dataset(Map, wfparams.altitudes, wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
    return WindFarmState(x_acts, x_obs, y_obs, X_field, Y_field)
end

# function initialize_state(b0::WindFarmBelief)
#     """ Called only once when the initial state is created. Assumes knowledge at same points as the initial belief. Should only be used with greedy solver. """
#     x_obs = b0.gpla_wf.x
#     y_hat = b0.gpla_wf.y
#     x_acts = b0.x_acts
#     return WindFarmState(x_acts, x_obs, y_hat)
# end

function POMDPs.rand(rng::AbstractRNG, b::WindFarmBelief)
    """ Called everytime a state is sampled from the belief. """
    y_hat = rand(b.gpla_wf)
    x_obs = b.gpla_wf.x
    x_acts = b.x_acts
    x_obs_full = reshape(Float64[],3,0)
    y_obs_full = Float64[]  
    return WindFarmState(x_acts, x_obs, y_hat, x_obs_full, y_obs_full)
end