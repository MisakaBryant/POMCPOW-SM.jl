using POMCPOW
using Test

using POMDPs
using POMDPModels
using ParticleFilters
using POMDPTools
using D3Trees
using RockSample

import Random

# 扩展 LightDark1D 的函数
POMDPs.states(::LightDark1D) = [LightDark1DState(x, y) for x in -10:10, y in -10:10]
POMDPs.stateindex(::LightDark1D, s::LightDark1DState) = Int(s.y + 11)
POMDPs.isterminal(::LightDark1D, s::LightDark1DState) = s.status < 0
POMDPs.support(::POMDPModels.LDNormalStateDist) = states(LightDark1D())


@testset "all" begin

    @testset "POMDPTesting" begin
        solver = POMCPOWSolver()
        pomdp = LightDark1D()
        test_solver(solver, pomdp, updater=DiscreteUpdater(pomdp))
        test_solver(solver, pomdp)

        solver = POMCPOWSolver(max_time=0.1, tree_queries=typemax(Int))
        test_solver(solver, pomdp, updater=DiscreteUpdater(pomdp))
    end

    
end;
