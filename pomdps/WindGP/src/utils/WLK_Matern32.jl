"""
Wind Log Kernel
Custom kernel for the wind farm domain.
This version uses Matern32 as the x-y kernel.
"""

mutable struct WLK_Matern32{T<:Real} <: GaussianProcesses.Isotropic{GaussianProcesses.SqEuclidean}
    # Matern32Kernel
    ℓ2_mat::T      # lengthscale
    σ2_mat::T      # signal variance

    # LinearExponentialKernel
    ℓ_lin::T      # lengthscale
    σ2_lin::T     # signal variance

    # WindLogLawKernel
    d::T          # zero-plane displacement
    zₒ::T         # roughness length

    # Priors for kernel parameters
    priors::Array
end


"""
GaussianProcesses.cov(k::Kernel, X1::AbstractMatrix, X2::AbstractMatrix)
Create covariance matrix from kernel `k` and matrices of observations `X1` and `X2`, where
each column is an observation.
"""
function GaussianProcesses.cov(k::WLK_Matern32, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    cK = Array{promote_type(eltype(X1), eltype(X2))}(undef, nobs1, nobs2)
    GaussianProcesses.cov!(cK, k, X1, X2, data)
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::WLK_Matern32, X::AbstractMatrix, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())
    dim, nobs = size(X)
    (nobs,nobs) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) and X has size $(size(X))"))
    @inbounds for j in 1:nobs
        for i in 1:nobs
            cK[i,j] = GaussianProcesses.cov_ij(k, X, X, data, i, j, dim)
        end
    end
    return cK
end

function GaussianProcesses.cov!(cK::AbstractMatrix, k::WLK_Matern32, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())
    if X1 === X2
        return GaussianProcesses.cov!(cK, k, X1, data)
    end
    dim1, nobs1 = size(X1)
    dim2, nobs2 = size(X2)
    dim1==dim2 || throw(ArgumentError("X1 and X2 must have same dimension"))
    dim = size(X1, 1)
    (nobs1,nobs2) == size(cK) || throw(ArgumentError("cK has size $(size(cK)) X1 $(size(X1)) and X2 $(size(X2))"))
    @inbounds for i in 1:nobs1
        for j in 1:nobs2
            cK[i,j] = GaussianProcesses.cov_ij(k, X1, X2, data, i, j, dim)
        end
    end
    return cK
end

function GaussianProcesses.cov(k::WLK_Matern32, x1::AbstractVector, x2::AbstractVector, data::GaussianProcesses.KernelData=GaussianProcesses.EmptyData())

    ℓ2_mat, σ2_mat, ℓ_lin, σ2_lin, d, zₒ = k.ℓ2_mat, k.σ2_mat, k.ℓ_lin, k.σ2_lin, k.d, k.zₒ

    x_star = x1
    x = x2
    
    z_star = x_star[3]
    z = x[3]

    k_val = 1        
    
    # Matern32Kernel
    r = x[1:2] - x_star[1:2]
    r_mat = dot_product(r,r)
    s = √3 * r_mat / ℓ2_mat;
    k_val *= σ2_mat * (1 + s) * exp(-s)

    # LinearExponentialKernel
    r = abs(z - z_star)
    k_val *= σ2_lin * exp(-r/(2*ℓ_lin))

    # WindLogLawKernel
    ratio = log((z_star-d)/zₒ) / log((z-d)/zₒ)
    # k_val *= (ratio - m[z_star]/uₓ) / (1 - m[z]/uₓ)
    k_val *= ratio

    return k_val
end

@inline GaussianProcesses.cov_ij(k::WLK_Matern32, X1::AbstractMatrix, X2::AbstractMatrix, i::Int, j::Int, dim::Int) = GaussianProcesses.cov(k, @view(X1[:,i]), @view(X2[:,j]))
@inline GaussianProcesses.cov_ij(k::WLK_Matern32, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.IsotropicData, i::Int, j::Int, dim::Int) = GaussianProcesses.cov_ij(k, X1, X2, i, j, dim)
@inline GaussianProcesses.cov_ij(k::WLK_Matern32, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.KernelData, i::Int, j::Int, dim::Int) = GaussianProcesses.cov_ij(k, X1, X2, i, j, dim)

WLK_Matern32(ll_mat::T, lσ_mat::T, ll_lin::T, lσ_lin::T, d::T, zₒ::T) where T = WLK_Matern32{T}(exp(ll_mat), exp(2 * lσ_mat), exp(ll_lin), exp(2 * lσ_lin), d, zₒ, [])

GaussianProcesses.num_params(se::WLK_Matern32) = 6

function GaussianProcesses.set_params!(se::WLK_Matern32, hyp::AbstractVector)
    length(hyp) == GaussianProcesses.num_params(se) || throw(ArgumentError("Should have retrieved $(GaussianProcesses.num_params(se)), but received $(length(hyp))."))
    ll_mat, lσ_mat, ll_lin, lσ_lin, d, zₒ = hyp   # extract hyp to vars
    se.ℓ2_mat, se.σ2_mat, se.ℓ_lin, se.σ2_lin, se.d, se.zₒ = exp(ll_mat), exp(2 * lσ_mat), exp(ll_lin), exp(2 * lσ_lin), d, zₒ
end

GaussianProcesses.get_params(se::WLK_Matern32{T}) where T = T[log(se.ℓ2_mat), log(se.σ2_mat) / 2, log(se.ℓ_lin) / 2, log(se.σ2_lin) / 2, se.d, se.zₒ]

GaussianProcesses.get_param_names(se::WLK_Matern32) = [:ll_mat, :lσ_mat, :ll_lin, :lσ_lin, :d, :zₒ]





# Derivative functions for first-order optimization algorithms.

@inline function dk_dll_mat(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector)
    r = x1[1:2] - x2[1:2]
    r_mat = dot_product(r,r)
    
    drv = r_mat / se.ℓ2_mat
    return drv * GaussianProcesses.cov(se, x1, x2)
end

@inline dk_dlσ_mat(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector) = 2 * GaussianProcesses.cov(se, x1, x2)

@inline function dk_dll_lin(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector)
    r = abs(x1[3] - x2[3])
    
    drv = r/se.ℓ_lin
    return drv * GaussianProcesses.cov(se, x1, x2)
end

@inline dk_dlσ_lin(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector) = 2 * GaussianProcesses.cov(se, x1, x2)

@inline function dk_dd(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector)
    z1, z2 = x1[3], x2[3]

    drv1 = log((z2 - se.d)/se.zₒ) * (z2 - se.d)
    drv2 = log((z1 - se.d)/se.zₒ) * (z1 - se.d)
    
    cov_val = GaussianProcesses.cov(se, x1, x2)

    return cov_val / drv1 - cov_val / drv2
end

@inline function dk_dzₒ(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector)
    z1, z2 = x1[3], x2[3]

    drv1 = log((z2 - se.d)/se.zₒ) * se.zₒ
    drv2 = log((z1 - se.d)/se.zₒ) * se.zₒ
    
    cov_val = GaussianProcesses.cov(se, x1, x2)

    return cov_val / drv1 - cov_val / drv2
end

@inline function GaussianProcesses.dKij_dθp(se::WLK_Matern32, X1::AbstractMatrix, X2::AbstractMatrix, data::GaussianProcesses.IsotropicData, i::Int, j::Int, p::Int, dim::Int)
    x1, x2 = @view(X1[:,i]), @view(X2[:,j])
    return dk_dθp(se, x1, x2, p)
end

# Retrieves derivative of cov function w.r.t. param index `p`.
@inline function dk_dθp(se::WLK_Matern32, x1::AbstractVector, x2::AbstractVector, p::Int)

    if p==1
        return dk_dll_mat(se, x1, x2)

    elseif p==2
        return dk_dlσ_mat(se, x1, x2)
    
    elseif p==3
        return dk_dll_lin(se, x1, x2)

    elseif p==4
        return dk_dlσ_lin(se, x1, x2)
    
    elseif p==5
        return dk_dd(se, x1, x2)

    elseif p==6
        return dk_dzₒ(se, x1, x2)

    else
        return NaN
    end
end

dk_dθp(se::WLK_Matern32, r::Real, p::Int) = error("This call should not have been made.")

GaussianProcesses.cov(se::WLK_Matern32, r::Number) = error("This call should not have been made.")
