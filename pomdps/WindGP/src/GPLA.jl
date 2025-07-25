# mutable struct GPLA{X<:AbstractMatrix, Y<:AbstractVector, M<:GaussianProcesses.Mean, K<:GaussianProcesses.Kernel, NOI<:GaussianProcesses.Param} <: GaussianProcesses.GPBase
mutable struct GPLA{M<:GaussianProcesses.Mean, K<:GaussianProcesses.Kernel, NOI<:GaussianProcesses.Param} <: GaussianProcesses.GPBase
    x::Matrix{Float64}
    y::Vector{Float64}
    k::Int64 #Local Neighborhood Size

    action_dims::Int64
    state_dims::Int64
    dim::Int64
    mean::M
    kernel::K
    logNoise::NOI

    kdtree::Union{Nothing, KDTree}

    "Auxiliary variables used to optimize GP hyperparameters"
    data::GaussianProcesses.KernelData
    mll::Float64
    dmll::Vector{Float64}
    target::Float64
    dtarget::Vector{Float64}
    # function GPLA{X, Y, M, K, NOI}(x::X, y::Y, k::Int64, action_dims::Int64, state_dims::Int64, mean::M, kernel::K, logNoise::NOI) where {X, Y, M, K, NOI}
    function GPLA{M, K, NOI}(x, y, k::Int64, action_dims::Int64, state_dims::Int64, mean::M, kernel::K, logNoise::NOI) where {M, K, NOI}
        data = GaussianProcesses.KernelData(kernel, x, x)
        gp = new{M, K, NOI}(ElasticArray(x), ElasticArray(y), k, action_dims, state_dims, action_dims+state_dims, mean, kernel, logNoise, nothing, data)
        initialize!(gp)
    end
end

function GPLA(x::AbstractMatrix, y::AbstractVector, k::Integer, action_dims::Integer, state_dims::Integer, mean::GaussianProcesses.Mean, kernel::GaussianProcesses.Kernel, logNoise::Real)
    lns = GaussianProcesses.wrap_param(logNoise)
    GPLA{typeof(mean),typeof(kernel), typeof(lns)}(x, y, k, action_dims, state_dims, mean, kernel, lns)
    # GPLA{typeof(x),typeof(y),typeof(mean),typeof(kernel), typeof(lns)}(x, y, k, action_dims, state_dims, mean, kernel, lns)
end

function initialize!(gp::GPLA)

    srt = sortperm(gp.x[end, :])   # sort by altitude to prevent non-PSD.
    gp.x = gp.x[:,srt]
    gp.y = gp.y[srt]

    n_obs = size(gp.y, 1)
    if n_obs != 0
        gp.kdtree = KDTree(gp.x)
        update_mll!(gp)
    end
    return gp
end

function GaussianProcesses.predict_f(gp::GPLA, x::AbstractArray{T,2} where T)
    nx = size(gp.x, 2)
    cols_x = eachcol(gp.x)
    if nx <= gp.k
        mx = GaussianProcesses.mean(gp.mean, gp.x)
        mf = GaussianProcesses.mean(gp.mean, x)
        Kxf = GaussianProcesses.cov(gp.kernel, x, gp.x) #size(size(x,2) x nx)
        Kff = GaussianProcesses.cov(gp.kernel, x, x) .+ exp(2*gp.logNoise.value) .+ eps()
        y = gp.y - mx
        data = GaussianProcesses.KernelData(gp.kernel, gp.x, gp.x)
        Σ = GaussianProcesses.cov(gp.kernel, gp.x, gp.x, data) + Matrix(I, nx, nx).*(exp(2*gp.logNoise.value)+eps())
        cK = PDMat(GaussianProcesses.make_posdef!(Σ)...)
        α = reshape(cK \ y, nx, 1)
        β = cK \ transpose(Kxf)
        μ = mf + Kxf*α
        σ² = diag(Kff - Kxf*β)
    else
        neighbors, _ = knn(gp.kdtree, extract_value(x), gp.k, true)
        μ = zeros(eltype(x), size(x, 2), 1)
        σ² = zeros(eltype(x), size(x, 2), 1)
        for i = 1:size(x,2)
            x_obs = gp.x[:, neighbors[i]]
            y_obs = gp.y[neighbors[i]]
            ##### Deploy this way for memoization (doesn't speed up) #####
            # mx, mf, Kxx, Kxf, s = predict_local(x[:,i:i], x_obs, gp.mean, gp.kernel, gp.logNoise)
            # y_obs = reshape(y_obs - mx, size(y_obs, 1), 1)
            # m = mf + GaussianProcesses.dot(Kxf, Kxx \ y_obs)
            #####
            m, s = predict_local(x[:,i:i], x_obs, y_obs, gp.mean, gp.kernel, gp.logNoise)

            # Enforce that locations in gp.x have zero variance.
            if x[:,i] in cols_x 
                s = 0.0
            end

            μ[i, 1] = m
            σ²[i, 1] = s #+ exp(2*gp.logNoise.value) + eps()
        end
    end

    return μ, clamp.(σ², 0.0, Inf)
end

function predict_local(x, x_obs, y_obs, mean, kernel, logNoise)
    k = size(x_obs, 2)

    sort_obs = sortperm(x_obs[end, :])   # sort by altitude to prevent non-PSD.
    x_obs = x_obs[:, sort_obs]
    y_obs = y_obs[sort_obs]

    mx = GaussianProcesses.mean(mean, x_obs)
    mf = GaussianProcesses.mean(mean, x)[1]
    Kxf = GaussianProcesses.cov(kernel, x, x_obs) #size(size(x,2) x nx)
    Kff = GaussianProcesses.cov(kernel, x, x) .+ exp(2*logNoise.value) .+ eps()

    y_obs = reshape(y_obs - mx, size(y_obs, 1), 1)
    data = GaussianProcesses.KernelData(kernel, x_obs, x_obs)
    Σ = GaussianProcesses.cov(kernel, x_obs, x_obs, data) + Matrix(I, k, k).*(exp(2*logNoise.value)+eps())
    Kxx = PDMat(GaussianProcesses.make_posdef!(Σ)...)
    μ = mf + GaussianProcesses.dot(Kxf, Kxx \ y_obs)
    Σ = Kff - Kxf*(Kxx \ transpose(Kxf))
    σ² = abs(Σ[1])
    return μ, σ²
end

function mll_local(idx, gp, mx, neighbors)
    if idx in neighbors
        neighbors = neighbors[neighbors .!= idx]
    else
        neighbors = neighbors[1:end-1]
    end
    k = length(neighbors)


    x = gp.x[:,idx:idx]
    x_obs = gp.x[:, neighbors]

    sort_neighs = sortperm(x_obs[end, :])   # sort by altitude to prevent non-PSD.
    neighbors = neighbors[sort_neighs]
    x_obs = gp.x[:, neighbors]

    y_obs = gp.y[neighbors] - mx[neighbors]
    y_obs = reshape(y_obs, size(y_obs, 1), 1)

    mf = mx[idx]
    Kxf = GaussianProcesses.cov(gp.kernel, x, x_obs) #size(size(x,2) x nx)
    Kff = GaussianProcesses.cov(gp.kernel, x, x) .+ exp(2*gp.logNoise.value) .+ eps()

    data = GaussianProcesses.KernelData(gp.kernel, x_obs, x_obs)
    Σ = GaussianProcesses.cov(gp.kernel, x_obs, x_obs, data) + Matrix(I, gp.k, gp.k).*(exp(2*gp.logNoise.value)+eps())
    Kxx = PDMat(GaussianProcesses.make_posdef!(Σ)...)
    μ = mf + GaussianProcesses.dot(Kxf, Kxx \ y_obs)
    Σ = Kff - Kxf*(Kxx \ transpose(Kxf))
    σ² = abs(Σ[1])
    σ = sqrt(σ²)
    log_p = -0.5*((gp.y[idx] - μ)/σ)^2 - 0.5*log(2*pi) - log(σ)
    param_tuple = (μ = μ, σ²  = σ², Kxx = Kxx, Kxf = Kxf, Kff = Kff, mf = mf, y = y_obs, neighbors = neighbors)
    return log_p, param_tuple
end

function update_mll!(gp::GPLA)
    nx = size(gp.x, 2)
    μ = GaussianProcesses.mean(gp.mean, gp.x)
    if nx <= gp.k
        y = gp.y - μ
        data = GaussianProcesses.KernelData(gp.kernel, gp.x, gp.x)
        Σ = GaussianProcesses.cov(gp.kernel, gp.x, gp.x, data) + Matrix(I, nx, nx).*(exp(2*gp.logNoise.value)+eps())
        cK = PDMat(GaussianProcesses.make_posdef!(Σ)...)
        α = cK \ y
        gp.mll = - (GaussianProcesses.dot(y, α) + GaussianProcesses.logdet(cK) + log(2*pi) * nx) / 2
    else
        mx = GaussianProcesses.mean(gp.mean, gp.x)
        neighbors, _ = knn(gp.kdtree, extract_value(gp.x), gp.k + 1)
        gp.mll = 0.0
        for i = 1:nx
            log_p, _ = mll_local(i, gp, mx, neighbors[i])
            gp.mll += log_p
        end
    end
end

"""
     update_dmll!(gp::GPE, ...)
Update the gradient of the marginal log-likelihood of Gaussian process `gp`.
"""
function update_dmll!(gp::GPLA;
                    noise::Bool=true, # include gradient component for the logNoise term
                    domean::Bool=true, # include gradient components for the mean parameters
                    kern::Bool=true, # include gradient components for the spatial kernel parameters
                    )
    n_mean_params = GaussianProcesses.num_params(gp.mean)
    n_kern_params = GaussianProcesses.num_params(gp.kernel)
    gp.dmll = zeros(Float64, noise + domean * n_mean_params + kern * n_kern_params)
    nobs = size(gp.y, 1)
    μ = GaussianProcesses.mean(gp.mean, gp.x)
    neighbors, _ = knn(gp.kdtree, gp.x, gp.k + 1)
    for i = 1:nobs
        _, params = mll_local(i, gp, μ, neighbors[i])
        d_dmll = zeros(Float64, noise + domean * n_mean_params + kern * n_kern_params)
        dmll_local!(d_dmll, gp, i, n_mean_params, n_kern_params, params; noise = noise, domean = domean, kern = kern)
        gp.dmll += d_dmll
    end
end

function dmll_local!(dmll::AbstractVector, gp::GPLA, idx::Int64, n_mean_params, n_kern_params, params;
                noise::Bool=true, domean::Bool=true, kern::Bool=true)
    i=1
    if noise
        @assert GaussianProcesses.num_params(gp.logNoise) == 1
        dmll[i] = dmll_noise(gp, idx, params)
        i += 1
    end
    if domean && n_mean_params>0
        dmll_m = @view(dmll[i:i+n_mean_params-1])
        dmll_mean!(dmll_m, gp, idx, params)
        i += n_mean_params
    end
    if kern
        dmll_k = @view(dmll[i:end])
        dmll_kern!(dmll_k, gp, idx, params)
    end
    return dmll
end

function dmll_noise(gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    Knoise = diagm(repeat([2*exp(2*gp.logNoise.value)], gp.k))
    σ = sqrt(params.σ²) + eps()
    dμ = -dot(params.Kxf, params.Kxx \ Knoise * (params.Kxx \ params.y))
    dσ = dot(params.Kxf[1,:], params.Kxx \ Knoise * (params.Kxx \ params.Kxf[1,:])) + 2*exp(2*gp.logNoise.value)
    dσ /= σ*2.0
    dlog_p = -(y - params.μ)/σ*(-dμ/σ - (y - params.μ)/params.σ²*dσ)
    dlog_p -= dσ/σ
    return dlog_p
end

function dmll_mean!(dmll::AbstractVector, gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    x = gp.x[:,idx:idx]
    d_mean = -ones(Float64, gp.k)
    σ = sqrt(params.σ²) + eps()
    dμ = GaussianProcesses.grad_stack(gp.mean, x)*(1.0 + GaussianProcesses.dot(params.Kxf, params.Kxx \ d_mean))
    dσ = [0.0]
    dlog_p = -(y - params.μ)/σ*(-dμ/σ - (y - params.μ)/params.σ²*dσ)
    dlog_p -= dσ/σ

    for i in 1:length(dlog_p)
        dmll[i] = dlog_p[i]
    end
    return dmll
end

function dmll_kern!(dmll::AbstractVector, gp::GPLA, idx::Int64, params::NamedTuple)
    y = gp.y[idx]
    x = gp.x[:,idx:idx]
    dKxx = grad_stack(gp.kernel, gp.x[:,params.neighbors], gp.x[:,params.neighbors])
    dKff = grad_stack(gp.kernel, x, x)[1, 1, :]
    dKxf = grad_stack(gp.kernel, x, gp.x[:,params.neighbors])[1,:,:]
    σ = sqrt(params.σ²) + eps()
    for i in 1:size(dKxf,2)
        dμ = GaussianProcesses.dot(dKxf[:,i], params.Kxx \ params.y)
        dμ -= GaussianProcesses.dot(params.Kxf, (params.Kxx \ dKxx[:,:,i]) * (params.Kxx \ params.y))
        dσ = dKff[i] - GaussianProcesses.dot(dKxf[:,i], params.Kxx \ params.Kxf[1,:])
        dσ += GaussianProcesses.dot(params.Kxf[1,:], (params.Kxx \ dKxx[:,:,i]) * (params.Kxx \ params.Kxf[1,:]))
        dσ -= GaussianProcesses.dot(params.Kxf[1,:], params.Kxx \ dKxf[:,i])
        dσ /= σ*2.0
        dlog_p = -(y - params.μ)/σ*(-dμ/σ - (y - params.μ)/params.σ²*dσ)
        dlog_p -= dσ/σ
        dmll[i] = dlog_p
    end
    return dmll
end

function update_mll_and_dmll!(gp::GPLA, precomp; kwargs...)
    update_mll!(gp)
    update_dmll!(gp; kwargs...)
end

function GaussianProcesses.update_target_and_dtarget!(gp::GPLA, precomp; params_kwargs...)
    update_mll_and_dmll!(gp, precomp; params_kwargs...)
    gp.target = gp.mll
    gp.dtarget = gp.dmll
end

function grad_stack(k::GaussianProcesses.Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
    data = GaussianProcesses.KernelData(k, X1, X2)
    nobs1 = size(X1, 2)
    nobs2 = size(X2, 2)
    stack = Array{eltype(X1)}(undef, nobs1, nobs2, GaussianProcesses.num_params(k))
    GaussianProcesses.grad_stack!(stack, k, X1, X2, data)
end

function extract_value(x::Array)
    x_out = zeros(Float64, size(x, 1), size(x, 2))
    for j = 1:size(x,2)
        for i = 1:size(x,1)
            if x[i, j] isa AbstractFloat
                x_out[i, j] = x[i, j]
            else
                x_out[i, j] = x[i, j].value
            end
        end
    end
    return x_out
end

GaussianProcesses.get_params_kwargs(::GPLA; kwargs...) = delete!(Dict(kwargs), :lik)

function GaussianProcesses.get_params(gp::GPLA; noise::Bool=true, domean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; append!(params, GaussianProcesses.get_params(gp.logNoise)); end
    if domean
        append!(params, GaussianProcesses.get_params(gp.mean))
    end
    if kern
        append!(params, GaussianProcesses.get_params(gp.kernel))
    end
    return params
end

function GaussianProcesses.optimize!(gp::GPLA; method = GaussianProcesses.LBFGS(), domean::Bool = true, kern::Bool = true,
                   noise::Bool = true, lik::Bool = true,
                   meanbounds = nothing, kernbounds = nothing,
                   noisebounds = nothing, likbounds = nothing, kwargs...)
    params_kwargs = GaussianProcesses.get_params_kwargs(gp; domean=domean, kern=kern, noise=noise, lik=lik)
    func = GaussianProcesses.get_optim_target(gp; params_kwargs...)
    init = GaussianProcesses.get_params(gp; params_kwargs...)  # Initial hyperparameter values
    opt_settings = Optim.Options(show_trace=true)
    try
        if meanbounds == kernbounds == noisebounds == likbounds == nothing
            results = Optim.optimize(func, init, method, opt_settings, kwargs...)
        else
            lb, ub = GaussianProcesses.bounds(gp, noisebounds, meanbounds, kernbounds, likbounds;
                            domean = domean, kern = kern, noise = noise, lik = lik)
            results = GaussianProcesses.optimize(func.f, func.df, lb, ub, init, Fminbox(method), opt_settings)
        end
        GaussianProcesses.set_params!(gp, Optim.minimizer(results); params_kwargs...)
        return results
    catch
        println("Gaussian Process Optimization Failed!")
        return nothing
    end
end

function GaussianProcesses.set_params!(gp::GPLA, hyp::AbstractVector;
                     noise::Bool=true, domean::Bool=true, kern::Bool=true)
    n_noise_params = GaussianProcesses.num_params(gp.logNoise)
    n_mean_params = GaussianProcesses.num_params(gp.mean)
    n_kern_params = GaussianProcesses.num_params(gp.kernel)

    i = 1
    if noise
        GaussianProcesses.set_params!(gp.logNoise, hyp[1:n_noise_params])
        i += n_noise_params
    end

    if domean && n_mean_params>0
        GaussianProcesses.set_params!(gp.mean, hyp[i:i+n_mean_params-1])
        i += n_mean_params
    end

    if kern
        GaussianProcesses.set_params!(gp.kernel, hyp[i:i+n_kern_params-1])
        i += n_kern_params
    end
    update_mll!(gp)
end

function GaussianProcesses.get_optim_target(gp::GPLA; params_kwargs...)
    function ltarget(hyp::AbstractVector)
        prev = GaussianProcesses.get_params(gp; params_kwargs...)
        try
            GaussianProcesses.set_params!(gp, hyp; params_kwargs...)
            # update_K!(gp)
            update_mll!(gp)
            return -gp.mll
        catch err
            # reset parameters to remove any NaNs
            GaussianProcesses.set_params!(gp, prev; params_kwargs...)

            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, LinearAlgebra.PosDefException)
                println(err)
                return Inf
            else
                # throw(err)
                return Inf
            end
        end
    end
    function ltarget_and_dltarget!(grad::AbstractVector, hyp::AbstractVector)
        prev = GaussianProcesses.get_params(gp; params_kwargs...)
        try
            GaussianProcesses.set_params!(gp, hyp; params_kwargs...)
            # update_K!(gp)
            update_mll!(gp)
            update_dmll!(gp; params_kwargs...)
            grad[:] = -gp.dmll
            return -gp.mll
        catch err
            # reset parameters to remove any NaNs
            GaussianProcesses.set_params!(gp, prev; params_kwargs...)
            if !all(isfinite.(hyp))
                println(err)
                return Inf
            elseif isa(err, ArgumentError)
                println(err)
                return Inf
            elseif isa(err, LinearAlgebra.PosDefException)
                println(err)
                return Inf
            else
                # throw(err)
                return Inf
            end
        end
    end

    function dltarget!(grad::AbstractVector, hyp::AbstractVector)
        ltarget_and_dltarget!(grad::AbstractVector, hyp::AbstractVector)
    end
    xinit = GaussianProcesses.get_params(gp; params_kwargs...)
    func = GaussianProcesses.OnceDifferentiable(ltarget, dltarget!, ltarget_and_dltarget!, xinit)
    return func
end

function GaussianProcesses.init_precompute(gp::GPLA) nothing end

function GaussianProcesses.fit!(gp::GPLA, x::AbstractArray, y::AbstractArray)
    length(y) == size(x,2) || throw(ArgumentError("Input and output observations must have consistent dimensions."))
    gp.x = x
    gp.y = y
    gp.data = GaussianProcesses.KernelData(gp.kernel, x, x)
    initialize!(gp)
end

GaussianProcesses.noise_variance(gp::GPLA) = noise_variance(gp.logNoise)

function getSparse_K(kernel, X1, X2)
    """get a smaller cK matrix for a single x point"""

    X1_length = size(X1,2)
    X2_length = size(X2,2)

    cK = Array{Float64}(undef, X1_length, X2_length)

    for i in 1:X1_length
        x1 = X1[:,i]

        for j in 1:X2_length
            x2 = X2[:,j]
            cK[i,j] = GaussianProcesses.cov(kernel, x1, x2)
        end
    end

    return cK
end


function GaussianProcesses.rand(gp::GPLA, Xs_gp::AbstractArray{T,2} where T)
    """ Randomly samples an entire farm, based on Sequential Gaussian Simulation """
    X_gp = gp.x
    numNeighbors = gp.k
    kernel = gp.kernel
    logNoise = GaussianProcesses.get_value(gp.logNoise)
    covstrat = GaussianProcesses.FullCovariance()

    if size(Xs_gp) == (3,)                  # if there is only one point, make AbstractArray 2 dimensional.
        Xs_gp = transform4GPjl([Xs_gp])
    end

    X_gp_set = Set(eachcol(X_gp))           # lookup in Set is O(1), we will take advantage of this.
    Xs_samples_val = Float64[]

    kdtree_X_gp = KDTree(X_gp)
    kdtree_Xs_gp = KDTree(Xs_gp)

    prequal_samples = Set{Int}()            # indices of previously sampled xs points. these should be unique w.r.t. points in X_gp.
    prequal_samples_val = Float64[]

    for (xs_idx, xs) in enumerate(eachcol(Xs_gp))

        neighbors_of_xs_in_Xs = []
        neighbors_of_xs_in_Xs_values = []

        try
            neighbors_of_xs_in_Xs, _ = knn(kdtree_Xs_gp, xs, numNeighbors)
            neighbors_of_xs_in_Xs = collect(intersect(prequal_samples, Set{Int}(neighbors_of_xs_in_Xs)))   # only take points in tree if they have been sampled earlier.

            neighbors_of_xs_in_Xs_values = prequal_samples_val[neighbors_of_xs_in_Xs]
        catch
            nothing
        end

        if length(kdtree_X_gp.data) <= numNeighbors
            neighbors_of_xs_in_X = collect(1:size(X_gp,2))      # all of the indices for points of X_gp, if the numNeighbors are more than the dataset size in X_gp
            neighbors_of_xs_in_X_values = gp.y
        else
            neighbors_of_xs_in_X, _ = knn(kdtree_X_gp, xs, numNeighbors)
            neighbors_of_xs_in_X_values = gp.y[neighbors_of_xs_in_X]
        end


        closest_neighbors_of_xs = hcat(Xs_gp[:,neighbors_of_xs_in_Xs], X_gp[:,neighbors_of_xs_in_X])
        closest_neighbors_of_xs_values = vcat(neighbors_of_xs_in_Xs_values, neighbors_of_xs_in_X_values)

        num_of_neigh = size(closest_neighbors_of_xs, 2)
        empty_cK = GaussianProcesses.alloc_cK(num_of_neigh)

        sort_neighs = sortperm(closest_neighbors_of_xs[end, :])   # sort by altitude to prevent non-PSD.
        X_active = closest_neighbors_of_xs[:, sort_neighs]


        data = GaussianProcesses.KernelData(kernel, X_active, X_active)
        K_xx = active_cK = GaussianProcesses.update_cK!(empty_cK, X_active, kernel, logNoise, data, covstrat)

        K_fx = getSparse_K(kernel, xs, X_active)
        K_xf = getSparse_K(kernel, X_active, xs)
        K_f = GaussianProcesses.cov(kernel, xs, xs)

        mf = mean(gp.mean, xs)
        mx = mean(gp.mean, closest_neighbors_of_xs)


        yx = closest_neighbors_of_xs_values[sort_neighs]

        if length(K_fx) == 0
            μ_star = mf    # there are no prior points
        else
            μ_star = mf + dot(K_fx, inv(K_xx.mat) * (yx - mx))
        end

        Σ_star = ones(1,1)*K_f                          # convert from Float64 to Array
        Lck = GaussianProcesses.whiten!(active_cK, K_xf)
        GaussianProcesses.subtract_Lck!(Σ_star, Lck)

        Σ_star = abs.(Σ_star[1]) + noise_variance(gp)   # mimics predict_y.
        σ_star = sqrt(Σ_star)    /10

        # Prevent sampling negative wind value
        σ_star > μ_star ? σ_star = abs(μ_star) : nothing

        xs_dist = Normal(μ_star, σ_star)
        xs_sampled_val = rand(xs_dist)
        push!(Xs_samples_val, xs_sampled_val)

        if !(xs in X_gp_set)                            # enforces uniqueness w.r.t. points in X_gp.
            push!(prequal_samples, xs_idx)
            push!(prequal_samples_val, xs_sampled_val)
        else
            push!(prequal_samples_val, NaN)
        end

    end
    return Xs_samples_val
end


GaussianProcesses.rand(gp::GPLA) = GaussianProcesses.rand(gp, gp.x)

# Converts AbstractArray{T,1} to AbstractArray{T,2}:
GaussianProcesses.predict_f(gp::GPLA, x::AbstractArray{T,1} where T) = GaussianProcesses.predict_f(gp, x[:,1:1])
