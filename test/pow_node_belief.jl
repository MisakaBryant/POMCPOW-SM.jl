using POMCPOW
using POMDPs
using SubHunt
using StaticArrays

# 定义一个简单的 POMDP 模型
struct SimplePOMDP <: POMDP{Float64, Int, Float64} end

# 定义观测权重函数（示例）
function obs_weight(model, s, a, sp, o)
    return pdf(Normal(sp, 0.2), o)  # 假设观测是一个正态分布
end

# 定义粒子和权重
particles = []  # 示例粒子和奖励
weights = []  # 权重
dist = CategoricalVector{Tuple{SubState, Float64}}(particles, weights)

# 创建 POWNodeBelief 实例
model = SubHuntPOMDP()
action = 1  # 动作
observation = SVector{8, Float64}(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)  # 观测

belief = POWNodeBelief{SubState, Int, SVector{8,Float64}, typeof(model)}(model, action, observation, dist)

# 打印结果
println("POWNodeBelief:", typeof(belief))
println("模型: ", typeof(belief.model))
println("动作: ", belief.a)
println("观测: ", belief.o)
println("分布: ", belief.dist)