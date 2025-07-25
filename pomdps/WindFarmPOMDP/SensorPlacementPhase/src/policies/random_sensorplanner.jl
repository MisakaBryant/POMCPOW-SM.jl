"""
    RandomPlanner
"""

struct RandomPlanner end

# Constructor
RandomPlanner(pomdp::WindFarmPOMDP, extra_params::Vector) = RandomPlanner()


function get_solution(pomdp::WindFarmPOMDP, solver::RandomPlanner)
    no_of_sensors = pomdp.timesteps
    X_field = CartIndices_to_Array(actions(pomdp))

    locs = rand(1:size(X_field, 2), no_of_sensors)
    x_sensors = X_field[:, locs]
    return x_sensors
end

get_solution(s0::WindFarmState, b0::WindFarmBelief, pomdp::WindFarmPOMDP, tlparams, wfparams, solver::RandomPlanner) = get_solution(pomdp, solver)