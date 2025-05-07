using POMDPs
using BasicPOMCP
using POMDPModels
using POMDPTools
using POMCPOW


function trail(policy::Policy, pomdp::POMDP; max_steps=100)
    hr = HistoryRecorder(max_steps=max_steps)
    rhist = simulate(hr, pomdp, policy)
    # for (s, b, a, r, sp, o) in rhist
    #     @show s, a, r, sp
    # end

    return discounted_reward(rhist)
end

function solve_and_evaluate(solver::AbstractPOMCPSolver, pomdp::POMDP; max_steps=100, total=1000)
    policy = solve(solver, pomdp)    

    r = 0.0

    for i in 1:total
        total_reward = trail(policy, pomdp, max_steps=max_steps)
        r += total_reward
    end

    avg_total_reward = r / total

    return avg_total_reward
end


tree_queries_list = [10, 20, 50, 100, 200, 500, 1000, 2000]
total_reward_list = []
for tree_queries in tree_queries_list
    # solver = POMCPSolver(tree_queries=tree_queries)
    solver = POMCPOWSolver(tree_queries=tree_queries)
    pomdp = LightDark1D()
    ADR = solve_and_evaluate(solver, pomdp)
    println("Total reward with $tree_queries tree queries: ", ADR)
    push!(total_reward_list, ADR)
end
