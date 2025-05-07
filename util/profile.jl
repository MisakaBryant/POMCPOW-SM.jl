using POMDPs
using POMCPOW
using ProfileView
using POMDPModels
using Random
using Profile
using ProfileView

#=
using Gallium
breakpoint(Pkg.dir("POMCPOW", "src", "solver.jl"), 40)
=#

solver = POMCPOWSolver(tree_queries=50_000,
                     eps=0.01,
                     enable_action_pw=false,
                     alpha_observation=1/8,
                     rng=MersenneTwister(2))

problem = LightDark1D()
policy = solve(solver, problem)
ib = initialstate(problem)
a = action(policy, ib)

@time a = action(policy, ib)

Profile.clear()
@profile a = action(policy, ib)
ProfileView.view()
