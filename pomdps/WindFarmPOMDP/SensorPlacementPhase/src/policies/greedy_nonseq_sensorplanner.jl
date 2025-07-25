"""
    GrdyNonSeqPlanner
    A non sequential planner that takes the greediest actions in a single step.
"""

struct GrdyNonSeqPlanner end

# Constructor
GrdyNonSeqPlanner(pomdp::WindFarmPOMDP, extra_params::Vector) = GrdyNonSeqPlanner()


function get_solution(b0::WindFarmBelief, pomdp::WindFarmPOMDP, solver::GrdyNonSeqPlanner)
    no_of_sensors = pomdp.timesteps
    best_actions = reshape(Float64[], 3, 0)

    b_last = b0
    
    while no_of_sensors > 0
        legal_actions = CartIndices_to_Array(actions(pomdp, b_last))

        gpla_wf = b_last.gpla_wf
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, legal_actions)
        σ = sqrt.(σ²)
        N = max(1, length(gpla_wf.y))
    
        z_value = 1.645   # chosen: 90 percent confidence interval
        UCB = μ + z_value / sqrt(N) * σ
    
        
        this_val = argmaxall(vec(UCB); threshold = 1e-6)
        this_action = legal_actions[:, rand(this_val)]

        best_actions = hcat(best_actions, this_action)

        b_last = WindFarmBelief(best_actions, b_last.gpla_wf)    # the belief over the wind field is the same, but the already selected locations are added to update the `legal_actions` inside the loop.
        no_of_sensors = no_of_sensors - 1
    end
    
    return best_actions
end

get_solution(s0::WindFarmState, b0::WindFarmBelief, pomdp::WindFarmPOMDP, tlparams, wfparams, solver::GrdyNonSeqPlanner) = get_solution(b0, pomdp, solver)