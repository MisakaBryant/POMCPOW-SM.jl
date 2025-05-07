struct POWNodeBelief{S,A,O,P}
    model::P
    a::A # may be needed in push_weighted! and since a is constant for a node, we store it
    o::O
    dist::CategoricalVector{Tuple{S,Float64}}

    POWNodeBelief{S,A,O,P}(m,a,o,d) where {S,A,O,P} = new(m,a,o,d)
    function POWNodeBelief{S, A, O, P}(m::P, s, a, sp, o, r) where {S, A, O, P}
        cv = CategoricalVector((convert(S, sp), convert(Float64, r)),
                                                 obs_weight(m, s, a, sp, o))
        new(m, a, o, cv)
    end
end

function POWNodeBelief(model::POMDP{S,A,O}, s, a, sp, o, r) where {S,A,O}
    POWNodeBelief{S,A,O,typeof(model)}(model, s, a, sp, o, r)
end

rand(rng::AbstractRNG, b::POWNodeBelief) = rand(rng, b.dist)
state_mean(b::POWNodeBelief) = first_mean(b.dist)
POMDPs.currentobs(b::POWNodeBelief) = b.o
POMDPs.history(b::POWNodeBelief) = tuple((a=b.a, o=b.o))


struct POWNodeFilter end

belief_type(::Type{POWNodeFilter}, ::Type{P}) where {P<:POMDP} = POWNodeBelief{statetype(P), actiontype(P), obstype(P), P}

init_node_sr_belief(::POWNodeFilter, p::POMDP, s, a, sp, o, r) = POWNodeBelief(p, s, a, sp, o, r)

function push_weighted!(b::POWNodeBelief, ::POWNodeFilter, s, sp, r)
    # Z(o | s, a, sp )
    w = obs_weight(b.model, s, b.a, sp, b.o)
    insert!(b.dist, (sp, convert(Float64, r)), w)
end

struct StateBelief{SRB<:POWNodeBelief}
    sr_belief::SRB
end

rand(rng::AbstractRNG, b::StateBelief) = first(rand(rng, b.sr_belief))
mean(b::StateBelief) = state_mean(b.sr_belief)
POMDPs.currentobs(b::StateBelief) = currentobs(b.sr_belief)
POMDPs.history(b::StateBelief) = history(b.sr_belief)

##################### 粒子滤波器 #####################

# 定义预测模型
struct MyPredictModel{S, A, O}
    problem::POMDP{S, A, O}
end

function ParticleFilters.predict!(pm::Vector{Tuple{S,Float64}}, model::MyPredictModel{S, A, O}, b, a, rng) where {S, A, O}
    for i in 1:length(pm)
        s = b.particles[i][1]  # 获取当前粒子状态
        if isterminal(model.problem, s)
            pm[i] = (s, 0.0)  # 如果状态是终止状态，返回当前状态和奖励0
            continue
        end
        # 生成下一个状态和奖励
        sp, o, r = @gen(:sp, :o, :r)(model.problem, s, a, rng)
        pm[i] = (sp, r)
    end
end

function ParticleFilters.particle_memory(::MyPredictModel{S, A, O}) where {S, A, O}
    return Vector{Tuple{S,Float64}}()  # 返回一个空的粒子内存
end

# 定义重加权模型
struct MyReweightModel{S, A, O}
    problem::POMDP{S, A, O}
end

function ParticleFilters.reweight!(wm::Vector, model::MyReweightModel, b, a, pm, o, rng)
    for i in 1:length(wm)
        wm[i] = obs_weight(model.problem, b.particles[i], a, pm[i][1], o)  # 根据观测更新权重
    end
end

# 定义重采样器
struct MyResampler end

function ParticleFilters.resample(::MyResampler, b::WeightedParticleBelief, model::MyPredictModel, reweight_model::MyReweightModel, b_prev, a, o, rng)
    if sum(b.weights) == 0.0
        b.weights = ones(length(b.particles))  # 如果权重和为0，则将权重设置为全1，TODO:不正确的处理方法
    end
    # 使用重要性采样进行重采样
    resampled_particles = sample(rng, b.particles, Weights(b.weights), length(b.particles))
    return ParticleCollection(resampled_particles)
end


function belief_update(::POWNodeFilter, p::POMDP{S,A,O}, b::Union{POWNodeBelief, ParticleCollection}, a, o, n, rng) where {S,A,O}
    # 如果粒子数不为n，则进行重采样
    # if root_belief
    if isa(b, ParticleCollection)
        # particles::Vector{S}
        belief = ParticleCollection([(rand(rng, b), 0.0) for i in 1:n])
    else
        # particles::Vector{Tuple{S, Float64}
        belief = ParticleCollection(b.dist.items)
        if n_particles(belief) != n
            n_belief = ParticleCollection([rand(rng, belief) for i in 1:n])
            belief = n_belief
        end
    end
    
    pf = BasicParticleFilter(MyPredictModel(p), MyReweightModel(p), MyResampler(), n, rng)
    
    # basic 粒子滤波
    updated = ParticleFilters.update(pf, belief, a, o)
    
    # 返回全1的权重
    weights = ParticleFilters.weights(updated)
    # CategoricalVector中的权重以累积分布表示
    cdf = 1:length(weights)

    updated_dist = CategoricalVector{Tuple{S,Float64}}(updated.particles, cdf)

    return POWNodeBelief{S,A,O,typeof(p)}(p, a, o, updated_dist)
end
