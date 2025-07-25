"""
    GeneticPlanner
"""

@with_kw struct GeneticPlanner
    no_of_iterations = 3
    populationSize = 50
    crossoverRate = 0.8
    mutationRate = 0.05
end

# Constructor
GeneticPlanner(pomdp::WindFarmPOMDP, extra_params::Vector) = GeneticPlanner(parse.(Float64, extra_params)...)


function get_solution(s0::WindFarmState, pomdp::WindFarmPOMDP, tlparams, wfparams, solver::GeneticPlanner, layouttype)

    println("### Starting Solver ###")

    init_locs_sensors() = rand(1:size(X_field, 2), no_of_sensors)

    cons_sensors(x) = [0]

    function mutation_func_sensors!(x)
        x[:] = rand(1:size_X_field, no_of_sensors)
    end


    no_of_sensors = pomdp.timesteps
    X_field = CartIndices_to_Array(actions(pomdp))
    size_X_field = size(X_field, 2)

    lx = fill(1, no_of_sensors)
    ux = fill(size_X_field, no_of_sensors)
    tc = fill(Int, no_of_sensors)

    lc, uc = [0.0], [0.0]

    cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
    constraints = MixedTypePenaltyConstraints(PenaltyConstraints([1e3], cb, cons_sensors), tc)

    opts = Evolutionary.Options(iterations = Int(solver.no_of_iterations), abstol = 1e-5)

    mthd = GA(populationSize = Int(solver.populationSize),
              crossoverRate = solver.crossoverRate,
              mutationRate = solver.mutationRate,
              selection = Evolutionary.sus,
              crossover = Evolutionary.uniform,
              mutation = mutation_func_sensors!
    )

    obj_func = locs -> - get_layout_profit(s0, locs, X_field, tlparams, wfparams, solver, layouttype)      # Note the negative sign, since GA is a minimizer.

    GA_result = Evolutionary.optimize(obj_func,
                                      constraints,
                                      init_locs_sensors,
                                      mthd,
                                      opts
    )
    
    x_sensors = X_field[:, GA_result.minimizer]
    return x_sensors
end

function get_layout_profit(s0, locs, X_field, tlparams, wfparams, solver::GeneticPlanner, layouttype)
    x_sensors = X_field[:, locs]
    x_sensors_expanded = expand_action_to_below_altitudes.(Array_to_CartIndices(x_sensors), Ref(wfparams.altitudes))

    x_sensors_obs = CartIndices_to_Array(flatten(x_sensors_expanded))
    gpla_wf = get_GPLA_for_gen(s0.x_obs, s0.y_obs, wfparams)
    obs_y = rand(gpla_wf, x_sensors_obs)

    # Get next state
    sp_x_acts = x_sensors
    sp_x_obs  = x_sensors_obs
    sp_y_obs  = obs_y
    sp = WindFarmState(sp_x_acts, sp_x_obs, sp_y_obs, s0.x_obs_full, s0.y_obs_full)
    GaussianProcesses.fit!(gpla_wf, sp_x_obs, sp_y_obs)

    result = get_layout_profit(sp, gpla_wf, tlparams, wfparams, layouttype)
    @show locs, result
    return Int(round(result))    # Evolutionary.jl requires returning as Int in this configuration.
end

get_solution(s0::WindFarmState, pomdp::WindFarmPOMDP, tlparams, wfparams, solver::GeneticPlanner) = get_solution(s0, pomdp, tlparams, wfparams, solver, tlparams.layouttype)