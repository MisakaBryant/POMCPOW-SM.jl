using POMDPs
using POMDPTools
using Distributions
using Random
using LinearAlgebra

# import BOMCP.x2s, BOMCP.s2x

struct LunarLander <: POMDP{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    dt::Float64
    m::Float64 # 1000's kg
    I::Float64 # 1000's kg*m^2
    Q::Vector{Float64}
    R::Vector{Float64}
end

function LunarLander(;dt::Float64=0.5, m::Float64=1.0, I::Float64=10.0)
    Q = [0.0, 0.0, 0.0, 0.1, 0.1, 0.01]
    R = [1.0, 0.01, 0.1]
    return LunarLander(dt, m, I, Q, R)
end

struct LanderActionSpace
    min_lateral::Float64
    max_lateral::Float64
    max_thrust::Float64
    max_offset::Float64
    function LanderActionSpace()
        new(-10.0, 10.0, 15.0, 1.0)
    end
end

function Base.rand(as::LanderActionSpace)
    lateral_range = as.max_lateral - as.min_lateral
    f_x = rand()*lateral_range + as.min_lateral
    f_z = rand()*as.max_thrust
    offset = (rand()-0.5)*2.0*as.max_offset
    return [f_x, f_z, offset]
end

function Base.rand(rng::AbstractRNG, as::LanderActionSpace)
    lateral_range = as.max_lateral - as.min_lateral
    f_x = rand(rng)*lateral_range + as.min_lateral
    f_z = rand(rng)*as.max_thrust
    offset = (rand()-0.5)*2.0*as.max_offset
    return [f_x, f_z, offset]
end

function update_state(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG, σ::Float64=1.0)

    x = s[1]
    z = s[2]
    θ = s[3]
    vx = s[4]
    vz = s[5]
    ω = s[6]

    f_lateral = a[1]
    thrust = a[2]
    δ = a[3]

    fx = cos(θ)*f_lateral - sin(θ)*thrust
    fz = cos(θ)*thrust + sin(θ)*f_lateral
    torque = -δ*f_lateral

    ax = fx/m.m
    az = fz/m.m
    ωdot = torque/m.I

    ϵ = randn(rng, 3)*σ
    vxp = vx + ax*m.dt + ϵ[1]*0.1
    vzp = vz + (az - 9.0)*m.dt + ϵ[2]*0.1
    ωp = ω + ωdot*m.dt + ϵ[3]*0.01

    xp = x + vx*m.dt
    zp = z + vz*m.dt
    θp = θ + ω*m.dt

    sp = [xp, zp, θp, vxp, vzp, ωp]
    return sp
end

function get_observation(s::Vector{Float64}, a::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG,
                    σz::Float64=1.0, σω::Float64=0.01, σx::Float64=0.1)
    z = s[2]
    θ = s[3]
    ω = s[6]
    xdot = s[4]
    agl = z/cos(θ) + randn(rng)*σz
    obsω = ω + randn(rng)*σω
    obsxdot = xdot + randn(rng)*σx
    o = [agl, obsω, obsxdot]
    return o
end

# new
function POMDPs.observation(p::LunarLander, a::Vector{Float64}, sp::Vector{Float64})
    z = sp[2]
    θ = sp[3]
    ω = sp[6]
    xdot = sp[4]
    μ = [z/cos(θ), ω, xdot]
    Σ = Diagonal(p.R)
    return MvNormal(μ, Σ)
end


function get_reward(s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64}; dt::Float64=0.1) #TODO
    x = sp[1]
    z = sp[2]
    δ = abs(x)
    θ = abs(sp[3])
    vx = sp[4]
    vz = sp[5]

    if δ >= 15.0 || θ >= 0.5
        r = -1000.0
    elseif z <= 1.0
        r = -(δ + vz^2) + 100.0
    else
        r = -1.0*dt*2.0
    end
    return r
end

# new
function POMDPs.reward(p::LunarLander, s, a, sp)
    get_reward(s, a, sp, dt=p.dt)
end

function POMDPs.gen(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}, rng::AbstractRNG=Random.GLOBAL_RNG)
    sp = update_state(m, s, a, rng=rng)
    o = get_observation(sp, a, rng=rng)
    r = get_reward(s, a, sp, dt=m.dt)
    return (sp=sp, o=o, r=r)
end

POMDPs.actions(::LunarLander) = LanderActionSpace()
POMDPs.actiontype(::LunarLander) = Vector{Float64}
POMDPs.discount(::LunarLander) = 0.99

function POMDPs.initialstate(::LunarLander)
    μ = [0.0, 50.0, 0.0, 0.0, -10.0, 0.0]
    σ = [0.1, 0.1, 0.01, 0.1, 0.1, 0.01]
    σ = diagm(σ)
    return MvNormal(μ, σ)
end

function POMDPs.isterminal(::LunarLander, s::Vector{Float64})
    x = s[1]
    z = s[2]
    δ = abs(x)
    θ = abs(s[3])
    if δ >= 15.0 || θ >= 0.5 || z <= 1.0
        return true
    else
        return false
    end
end

function POMDPTools.obs_weight(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64}, o::Vector{Float64})
    R = [1.0, 0.01, 0.1]
    z = sp[2]
    θ = sp[3]
    ω = sp[6]
    xdot = s[4]
    agl = z/cos(θ)
    dist = MvNormal([agl, ω, xdot], R)
    return pdf(dist, o)
end

struct LanderPolicy <: Policy
    m::LunarLander
end

# POMDPs.updater(p::LanderPolicy) = EKFUpdater(p.m, p.m.Q.^2, p.m.R.^2)

function POMDPs.action(p::LanderPolicy, b::MvNormal)
    s = mean(b)
    act = [-0.5*s[4] -0.5*s[5] 0.0][1,:]
    return act
end

# x = s[1]
# z = s[2]
# θ = s[3]
# vx = s[4]
# vz = s[5]
# ω = s[6]

##### For EKF Belief Updater
# function BOMCP.x2s(m::LunarLander, x::Vector{Float64})
#     s = x
#     return s
# end

# function BOMCP.s2x(m::LunarLander, s::Vector{Float64})
#     x = s
#     return x
# end

# function BOMCP.gen_A(m::LunarLander, s::Vector{Float64}, a::Vector{Float64})
#     θ = s[3]
#     f_l = a[1]
#     thrust = a[2]
#     A = zeros(Float64, 6, 6)
#     A[1,1] = 1.0
#     A[1,4] = m.dt
#     A[2,2] = 1.0
#     A[2,5] = m.dt
#     A[3,3] = 1.0
#     A[3,6] = m.dt
#     A[4,3] = (-sin(θ)*f_l - cos(θ)*thrust)*m.dt/m.m
#     A[4,4] = 1.0
#     A[5,3] = (-sin(θ)*thrust + cos(θ)*f_l)*m.dt/m.m
#     A[5,5] = 1.0
#     A[6,6] = 1.0
#     return A
# end

# function BOMCP.gen_C(m::LunarLander, s::Vector{Float64})
#     z = s[2]
#     θ = s[3]
#     C = zeros(Float64, 3, 6)
#     C[1,2] = 1/(cos(θ) + eps())
#     C[1,3] = z*sin(θ)/(cos(θ)^2 + eps())
#     C[2,6] = 1.0
#     C[3,4] = 1.0
#     return C
# end
