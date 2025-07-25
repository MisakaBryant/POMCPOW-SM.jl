"""
    BayesianPlanner
"""

@with_kw struct BayesianPlanner
    no_of_iterations = 300
    every_n_samples = 20     # lower is more better, but more expensive.
end

# Constructor
BayesianPlanner(pomdp::WindFarmPOMDP, extra_params::Vector) = BayesianPlanner(parse.(Int, extra_params)...)


function get_solution(s0::WindFarmState, pomdp::WindFarmPOMDP, tlparams, wfparams, solver::BayesianPlanner, layouttype)
    
    println("### Starting Solver ###")

    no_of_sensors = pomdp.timesteps
    X_field = CartIndices_to_Array(actions(pomdp))
    size_X_field = size(X_field, 2)

    lx = fill(1, no_of_sensors)
    ux = fill(size_X_field, no_of_sensors)

    model = GaussianProcesses.ElasticGPE(no_of_sensors,
                                         mean = MeanConst(0.5),
                                         kernel = Mat52Ard(zeros(no_of_sensors), 0.),
                                         logNoise = -2.,
                                         capacity = 3000    # initial capacity of ElasticArray (pre-allocation)
    )

    modeloptimizer = BayesianOptimization.MAPGPOptimizer(every = solver.every_n_samples,
                                                         noisebounds = [-4, 3],
                                                         kernbounds = [[-3*ones(no_of_sensors); -3],
                                                                       [4*ones(no_of_sensors); 3]],
                                                         maxeval = 100
    )

    obj_func = locs -> get_layout_profit(s0, locs, X_field, tlparams, wfparams, solver, layouttype)

    opt = BayesianOptimization.BOpt(obj_func,                                   # objective function
                                    model,                                      # Gaussian model over objective function values
                                    ExpectedImprovement(),                      # type of acquisition
                                    modeloptimizer,                             # optimizer options
                                    lx,                                         # input lowerbounds
                                    ux,                                         # input upperbounds
                                    maxiterations = solver.no_of_iterations,    # max no of iterations
                                    sense = BayesianOptimization.Max,           # purpose [`Min`: minimizer, `Max`: maximizer]
                                    verbosity = BayesianOptimization.Silent,    # verbose intensity
    )

    result = BayesianOptimization.boptimize!(opt)
    best_locs = result.observed_optimizer
    x_sensors = X_field[:, Int.(round.(best_locs))]
    
    return x_sensors
end

function get_layout_profit(s0, locs, X_field, tlparams, wfparams, solver::BayesianPlanner, layouttype)
    locs = Int.(round.(locs))
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

    result = get_layout_profit(sp, gpla_wf, tlparams, wfparams, layouttype) / 1.0e7 / tlparams.no_of_turbines  # TODO.
    @show locs, result
    return result
end

get_solution(s0::WindFarmState, pomdp::WindFarmPOMDP, tlparams, wfparams, solver::BayesianPlanner) = get_solution(s0, pomdp, tlparams, wfparams, solver, tlparams.layouttype)