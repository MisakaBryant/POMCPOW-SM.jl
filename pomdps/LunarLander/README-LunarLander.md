### Partially Observable Lunar Lander（部分可观察月球着陆器）

#### 问题介绍

- 是著名的 Lunar Lander 问题的部分可观察变体。通过垂直和横向推力进行控制，如火焰所示。对高度的有噪声观测给出了沿着陆器垂直轴从中心到地面的距离，如红色虚线所示。
- 任务目标：引导着陆器在目标区域以低冲击力着陆。
- 着陆器的状态用六元组$(x,y,\theta,\dot x,\dot y,\omega)$表示，$x$ 水平位置、$y$ 垂直位置、$\theta$ 方位角、$\dot x$ 水平速度、$\dot y$ 垂直速度、$\omega$ 角速度。着陆器对角速度 $\omega$、水平速度 $\dot x$、高度（红色虚线）进行有噪声观测。动作空间是连续的，用三元组$(T,F_x,\delta)$表示，$T$ 表示沿着陆器垂直轴通过质心的主推力（$T \in [0,15]$）、$F_x$ 表示沿水平轴与质心偏离距离 $\delta$ 的修正推力（$F_x \in [-5,5]\ \ and \ \ \delta \in [-1,1]$）。
- 着陆器的初始状态从均值 $\mu=(x=0,y=50,\theta=0,\dot x=0,\dot y=-10,\omega=0)$ 的多元高斯分布中采样得到。
- 奖励函数定义如下：

$$
r(s,a,s') =
\left\{
\begin{array}{ll}
-1000, & \text{if } x \ge 15 \text{ or } \theta \ge 0.5 \\
100 - x - v_y^2, & \text{if } y \le 1 \\
-1, & \text{otherwise}
\end{array}
\right.
$$

- 第一项给出了在着陆器进入无可恢复状态时的惩罚，第二项给出了着陆的正奖赏、偏离中心点的惩罚、冲击速度的惩罚，第三项给出了燃料消耗的常数惩罚。
- 使用扩展的卡尔曼滤波器（EKF）进行更新。

#### 代码细节

- 结构体 LunarLander

```julia
struct LunarLander <: POMDP{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    dt::Float64         # 时间微分，默认值0.5
    m::Float64          # 着陆器质量，默认值1.0       1000's kg
    I::Float64          # 着陆器转动惯量，默认值10.0   1000's kg*m^2
    Q::Vector{Float64}  # 状态值的过程噪声 [0.0, 0.0, 0.0, 0.1, 0.1, 0.01]
    R::Vector{Float64}  # 观察值的测量噪声 [1.0, 0.01, 0.1]
end
```

- 结构体 LanderActionSpace，其中 max_offset 是单侧最大偏移，左右相加应为其两倍。

```julia
struct LanderActionSpace
	# 其中的垂直和横向都是相对着陆器而言，并非地面
    min_lateral::Float64
    max_lateral::Float64
    max_thrust::Float64
    max_offset::Float64
	# 文中数据是(-5.0, 5.0, 15.0, 1.0)
    function LanderActionSpace()
        new(-10.0, 10.0, 15.0, 1.0)
    end
end
```

- LanderActionSpace 为参数类型的随机函数，生成真正的动作，结构体只是动作的参数。

```julia
function Base.rand(as::LanderActionSpace)
    lateral_range = as.max_lateral - as.min_lateral
    f_x = rand()*lateral_range + as.min_lateral
    f_z = rand()*as.max_thrust
    offset = (rand()-0.5)*2.0*as.max_offset
    return [f_x, f_z, offset] # 横向推力，垂直推力，横向推力与质心的偏移
end
```

- 结构体 LanderPolicy

```julia
struct LanderPolicy <: Policy
    m::LunarLander
end
```

- update_state 方法，通过动作对状态更新，加速度确切，速度添加了高斯噪声，位置角度确切。

```julia
function update_state(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG, σ::Float64=1.0)
    # 相对地面的水平和竖直
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
```

- get_observation 方法（不知道垂直速度）

```julia
# 参数a（动作）未使用
function get_observation(s::Vector{Float64}, a::Vector{Float64}; rng::AbstractRNG=Random.GLOBAL_RNG, σz::Float64=1.0, σω::Float64=0.01, σx::Float64=0.1)
    z = s[2]
    θ = s[3]
    ω = s[6]
    xdot = s[4]
    agl = z/cos(θ) + randn(rng)*σz
    obsω = ω + randn(rng)*σω
    obsxdot = xdot + randn(rng)*σx
    o = [agl, obsω, obsxdot] # 有噪红色虚线方向距离，有噪角速度，有噪水平（相对地面）线速度
    return o
end
```

- get_reward 方法

```julia
function get_reward(s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64}; dt::Float64=0.1)
    x = sp[1]
    z = sp[2]
    δ = abs(x)
    θ = abs(sp[3])
    vx = sp[4]
    vz = sp[5]

    if δ >= 15.0 || θ >= 0.5     # 状态失控
        r = -1000.0
    elseif z <= 1.0              # 接近着陆
        r = -(δ + vz^2) + 100.0
    else
        r = -1.0*dt*2.0
    end
    return r
end
```

- 改写了 POMDPs 中的一些方法

```julia
function POMDPs.reward(p::LunarLander, s, a, sp)
    get_reward(s, a, sp, dt=p.dt)
end

function POMDPs.gen(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}, rng::AbstractRNG=Random.GLOBAL_RNG)
    sp = update_state(m, s, a, rng=rng)
    o = get_observation(sp, a, rng=rng)
    r = get_reward(s, a, sp, dt=m.dt)
    return (sp=sp, o=o, r=r)
end

function POMDPs.initialstate_distribution(::LunarLander)
    μ = [0.0, 50.0, 0.0, 0.0, -10.0, 0.0]
    σ = [0.1, 0.1, 0.01, 0.1, 0.1, 0.01]
    σ = diagm(σ)           # 矩阵对角线
    return MvNormal(μ, σ)  # 多元高斯分布
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

POMDPs.updater(p::LanderPolicy) = EKFUpdater(p.m, p.m.Q.^2, p.m.R.^2)
```

- obs_weight 方法

```julia
function POMDPModelTools.obs_weight(m::LunarLander, s::Vector{Float64}, a::Vector{Float64}, sp::Vector{Float64}, o::Vector{Float64})
    R = [1.0, 0.01, 0.1]
    z = sp[2]
    θ = sp[3]
    ω = sp[6]
    xdot = s[4]
    agl = z/cos(θ)
    dist = MvNormal([agl, ω, xdot], R)
    return pdf(dist, o) # 概率密度函数
end
```

