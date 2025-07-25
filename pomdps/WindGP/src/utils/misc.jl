# Some helper functions. Some simple functions are re-defined here to avoid dependencies to other modules.

argmaxall(A) = findall(A .== maximum(A))    # returns indices of all the argmaxing values.
argmaxall(A; threshold = eps()) = findall(maximum(A) .- A .<= threshold)    # returns indices of all the argmaxing values.

eye(x::Int) = [a==b ? 1.0 : 0.0 for a in 1:x, b in 1:x]   # create identity matrix (no dependencies).

nearestRound(x::Number,i) = (x % i) > (i/2) ? x + i - x%i : x - x%i   # rounds x to the nearest multiple of i.
nearestRound(x::AbstractArray,i) = nearestRound.(x,i)

tuple_to_array(T) = [item for item in T]

prob_normalize(v) = v ./ sum(v)

flatten(A::AbstractArray) = vcat(A...)

Int_round(x::Number) = Int(round(x))

function writedlm_append(fname, data)
    if isnothing(fname)
        @warn "Not saved to disk!"
        return
    else
        open(fname, "a") do io
            DelimitedFiles.writedlm(io, data)
        end
        println("### Saved to:  $fname ###")
    end
end

function CartIndices_to_Vector(a::CartesianIndex)
    a = collect(a.I)[:,1:1]    # Convert from CartesianIndex to Vector.
    a = Float64.(a)
    return a
end

function CartIndices_to_Array(A::Array{CartesianIndex{N},T} where {N,T})
    A = CartIndices_to_Vector.(A)
    return transform4GPjl(A)
end

Vector_to_CartIndices(a::AbstractVecOrMat) = CartesianIndex(Int.(a)...)
Array_to_CartIndices(A::AbstractArray) = Vector_to_CartIndices.(eachcol(A))

# Calling with methods below simply ignores the transformation request (since it is already in that form).
CartIndices_to_Vector(x::AbstractVecOrMat) = x
CartIndices_to_Array(x::AbstractArray) = x
Vector_to_CartIndices(x::CartesianIndex) = x
Array_to_CartIndices(x::Array{CartesianIndex{N},T} where {N,T}) = x

function maxk!(ix, a, k; initialized=false)         # picks top k values in an array. 
    partialsortperm!(ix, a, 1:k, rev=true, initialized=initialized)
    @views collect(zip(ix[1:k], a[ix[1:k]]))
end

function makeHermitian!(A; inflation=1e-6)
    A[:,:] = 0.5 .* (A + A')                        # average with transpose.
    A[:,:] = A[:,:] + inflation .* eye(size(A,1))   # prevent singularity.
end    

function dropBelowThreshold!(A; threshold=eps(Float64))
    for idx in eachindex(A)
        if abs(A[idx]) < threshold
           A[idx] = 0
        end
    end 
end

function transform4GPjl(a::AbstractArray{Float64,1})
    """ Transform single location """
    a_gp = Array{Float64,2}(undef, 3, 1)
    for (idx,val) in enumerate(a)
        a_gp[idx] = val
    end
    return a_gp
end

function transform4GPjl(X::AbstractArray)
    """ Transform AbstractArray of locations """
    X_gp = Array{Float64,2}(undef, 3, length(X))
    for idx in 1:size(X,1)
        X_gp[:,idx] = X[idx]
    end
    return X_gp
end

function exactLogLaw(z_star::Number, z::Number, zₒ, d)
    return log((z_star-d)/zₒ) / log((z-d)/zₒ)
end

function euclidean_dist(a::CartesianIndex,b::CartesianIndex)
    diff = abs(a-b).I
    return sqrt(diff[1]^2 + diff[2]^2)
end

function euclidean_dist(a::AbstractArray,b::AbstractArray)
    diff = abs.(a-b)
    dp = dot_product(diff, diff)
    return sqrt(dp)
end

function dot_product(a::Union{Number,AbstractArray}, b::Union{Number,AbstractArray})    # Supports numbers, vectors and matrices.
    @assert size(a) == size(b) "Dimensions of the two inputs do not match."
    c = 0
    for i in eachindex(a)
        c += a[i]*b[i]
    end
    return c
end

function dot_product(a::CartesianIndex, b::CartesianIndex)    # Supports numbers, vectors and matrices.
    @assert length(a) == length(b) "Dimensions of the two inputs do not match."    
    a, b = tuple_to_array.([a.I, b.I])
    return dot_product(a, b)
end

function stretch(a::AbstractArray, lower_lim::Real, upper_lim::Real)  # stretches (forces given extremas, interpolates others in-between) any-size Array.
    mini, maxi = minimum(a), maximum(a)
    norma = (maxi-mini)/(upper_lim-lower_lim)

    b = (a .- mini) ./ norma .+ lower_lim
    return b
end

function stretch!(a::AbstractArray, lower_lim::Real, upper_lim::Real)  # stretches (forces given extremas, interpolates others in-between) any-size Array in-place.
    mini, maxi = minimum(a), maximum(a)
    norma = (maxi-mini)/(upper_lim-lower_lim)

    b = (a .- mini) ./ norma .+ lower_lim

    for i = firstindex(a):lastindex(a)
        a[i] = b[i]
    end
end

function split_arrays_randomly(X::AbstractArray, Y::AbstractArray, ratio)
    sz = size(X,2)
    srt = shuffle(1:sz)
    X = X[:, srt]
    Y = Y[srt]
    return X[:, 1:Int_round(sz * ratio)], X[:, Int_round(sz * ratio) + 1 : end], Y[1:Int_round(sz * ratio)], Y[Int_round(sz * ratio) + 1 : end]
end

function deg2NSEW(x_deg,y_deg)
    # Converts degress to direction, degrees, minutes, seconds.
    # E.g. +43.040958∘, -77.241711∘  ->  N 43∘2’27.45”, W 77◦14’30.16”.
    x_NSEW = Dict()
    y_NSEW = Dict()

    if x_deg>=0
        push!(x_NSEW, (:dir=>'N'))
    else
        push!(x_NSEW, (:dir=>'S'))
    end
    
    if y_deg>=0
        push!(y_NSEW, (:dir=>'E'))
    else
        push!(y_NSEW, (:dir=>'W'))
    end

    function getNSEW(val)
        val = abs(val)
        deg = floor(val)
        min = floor(abs((val - deg) * 60))
        sec = (abs(val - deg) * 60 - min) * 60
        return Dict(:deg=>deg, :min=>min, :sec=>sec)
    end

    merge!(x_NSEW, getNSEW(x_deg))
    merge!(y_NSEW, getNSEW(y_deg))

    return x_NSEW, y_NSEW
end

function savefig_recursive(plt_obj, filename, reset; dir="Figures")
    
    # If the filename exists, add a number to it and then save. Prevents overwrite.
    # filename: Name of file. Don't put filetype (e.g. "png") to the end of filename! Defaults to ".png".
    # reset: Delete everything in folder before saving? Set to true or false.
    
    if reset
        run(`rm -rf ./$dir`)
        mkdir("$dir")  # recreate the folder.
        filedir = dir*"/"*filename
        savefig(plt_obj, filedir)
    else
        dir in readdir() ? nothing : mkdir("$dir")  # if the folder exists, do nothing, otherwise, create it.
        i = 1
        itemname = filename
        while true in [occursin(itemname, item) for item in readdir(dir)]
            itemname = filename*string(i)
            i += 1
        end
        filedir = dir*"/"*itemname
        savefig(plt_obj, filedir)
    end

    return nothing
end

