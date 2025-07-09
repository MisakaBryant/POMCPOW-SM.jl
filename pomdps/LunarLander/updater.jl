function POMDPs.updater(p::PAPOMCPOWPlanner)
    rng = MersenneTwister(rand(p.solver.rng, UInt32))
    return BootstrapFilter(p.problem, 10000, rng)
end
