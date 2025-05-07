function POMDPs.updater(p::POMCPOWPlanner)
    rng = MersenneTwister(rand(p.solver.rng, UInt32))
    return BootstrapFilter(p.problem, Int(min(10000, 10*p.solver.tree_queries, 1e7*p.solver.max_time)), rng)
end

