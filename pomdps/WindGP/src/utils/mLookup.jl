# Lookup mean function

"""
    MeanLookup <: GaussianProcesses.Mean
Lookup mean function
"""
mutable struct MeanLookup <: GaussianProcesses.Mean
    kdtree::NearestNeighbors.KDTree    # Lookup K-Dimensional Tree of X vals
    Y::Vector{Float64}          # Lookup Y vals
    num_of_neighbors::Int
    dist_degree::Int
    Dict_of_Lookup::Dict    # What we really want
end

# MeanLookup(X::AbstractMatrix{Float64}, Y::AbstractVector{Float64}, num_of_neighbors::Int=10, dist_degree::Int=2) = new(NearestNeighbors.KDTree(X), Y, num_of_neighbors, dist_degree)

function MeanLookup(X::AbstractMatrix{Float64}, Y::AbstractVector{Float64}, X_field::Matrix{Float64}, num_of_neighbors::Int=10, dist_degree::Int=2)
    kdtree = NearestNeighbors.KDTree(X)
    mLookup_temp = MeanLookup(kdtree, Y, num_of_neighbors, dist_degree, Dict())
    keysLookup = Array.(collect(eachcol(X_field)))   # convert from SubArray slice to Array
    valsLookup = mean_v2(mLookup_temp, X_field)
    return MeanLookup(kdtree, Y, num_of_neighbors, dist_degree, Dict(keysLookup .=> valsLookup))
end

GaussianProcesses.mean(mLookup::MeanLookup, x::AbstractVector) = mLookup.Dict_of_Lookup[x]
GaussianProcesses.mean(mLookup::MeanLookup, x::AbstractMatrix) = getindex.(Ref(mLookup.Dict_of_Lookup), collect(eachcol(x)))


""" 
    Previous versions of the mean function for this kernel
"""

function mean_v2(mLookup::MeanLookup, x::AbstractVector)
    nn, dd = knn(mLookup.kdtree, x, mLookup.num_of_neighbors)
    dd_degreed = inv.(dd.^mLookup.dist_degree)
    # return mLookup.Y[nn][1]
    mult = sum(mLookup.Y[nn] .* dd_degreed) / sum(dd_degreed)
    return mult
end

function mean_v2(mLookup::MeanLookup, x::AbstractMatrix)
    num_of_neighbors = mLookup.num_of_neighbors
    dist_degree = mLookup.dist_degree

    nn, dd = knn(mLookup.kdtree, x, num_of_neighbors)
    nns = [mLookup.Y[item] for item in nn]
    dds = [item.^dist_degree for item in dd]

    # global X = x
    # global MLookup = mLookup
    mult = Float64[sum(n.*inv.(d))/sum(inv.(d)) for (n,d) in zip(nns, dds)]
    return mult
    # dd_degreed = dd.^dist_degree
    # return mLookup.Y[nn]
end

function mean_v1(mLookup::MeanLookup, x::AbstractVector)
    nn, _ = knn(mLookup.kdtree, x, 1)
    return mLookup.Y[nn][1]
end

function mean_v1(mLookup::MeanLookup, x::AbstractMatrix)
    nn, _ = knn(mLookup.kdtree, x, 1)
    nn = [item for sublist in nn for item in sublist]    # extract list of lists
    return mLookup.Y[nn]
end

GaussianProcesses.get_params(mLookup::MeanLookup) = [mLookup.kdtree, mLookup.Y]
GaussianProcesses.get_param_names(::MeanLookup) = [:kdtree, :Y]
GaussianProcesses.num_params(mLookup::MeanLookup) = length(mLookup.Y)    # returns number of datapoints in Lookup

function GaussianProcesses.set_params!(mLookup::MeanLookup, hyp::AbstractVector)
    (isa(hyp[1], NearestNeighbors.KDTree) && isa(hyp[2], AbstractVector{Float64})) || throw(ArgumentError("Hyp params do not match MeanLookup format."))
    copyto!(mLookup.β, hyp)
end