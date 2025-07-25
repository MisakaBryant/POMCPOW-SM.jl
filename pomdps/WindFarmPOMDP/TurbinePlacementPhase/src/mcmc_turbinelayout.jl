"""
    MCMC method to heuristically determine optimal turbine layout.
"""

@with_kw struct MCMCTurbineLayout <: TurbineLayoutType 
    no_of_trials = 300
    no_of_iterations = 300
end

function get_turbine_layout(gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::MCMCTurbineLayout)

    no_of_turbines = tlparams.no_of_turbines
    X_field = CartIndices_to_Array(turbine_action_space(tlparams, wfparams))

    
    no_of_trials = layouttype.no_of_trials
    no_of_iterations = layouttype.no_of_iterations    # per trial
    obj_func = x -> turbine_approximate_profits(x, gpla_wf, tlparams)
    
    best_state = reshape(Float64[], 3, 0)
    best_state_val = -Inf
    
    while no_of_trials > 0
        
        init_state, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, GreedyTurbineLayout())
        next_state = mcmc(init_state,
                          obj_func,
                          no_of_iterations,
                          X_field,
                          tlparams
        )

        next_state_val = obj_func(next_state)
        best_state_val ≤ next_state_val ? (best_state, best_state_val) = (copy(next_state), copy(next_state_val)) : nothing

        # @show no_of_trials, obj_func(init_state), obj_func(best_state)
        no_of_trials = no_of_trials - 1
    end

    expected_revenue = obj_func(best_state)
    return best_state, expected_revenue
end