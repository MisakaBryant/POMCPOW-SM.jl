using ParticleFilters
using Random
using LinearAlgebra

# 创建一个粒子集合
particles = [1.0, 2.0, 3.0, 4.0]
pc = ParticleCollection(particles)

println("粒子集合: ", pc.particles)

# 使用默认随机数生成器采样
sampled_particle = rand(pc)
println("采样的粒子: ", sampled_particle)

# 使用指定的随机数生成器采样
rng = MersenneTwister(42)
sampled_particle = rand(rng, pc)
println("采样的粒子: ", sampled_particle)