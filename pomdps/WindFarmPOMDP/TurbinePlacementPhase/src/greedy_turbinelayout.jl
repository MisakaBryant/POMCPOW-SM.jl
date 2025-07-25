"""
    Greedy method to heuristically determine optimal turbine layout.
"""

struct GreedyTurbineLayout <: TurbineLayoutType end

function get_turbine_layout(gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::GreedyTurbineLayout)
    
    no_of_turbines = tlparams.no_of_turbines
    X_field = CartIndices_to_Array(turbine_action_space(tlparams, wfparams))

    x_turbines = reshape(Float64[], 3, 0)

    while no_of_turbines > 0
        X_field = remove_seperated_locations(X_field, x_turbines, tlparams)
        next_turbine = get_next_turbine_location(gpla_wf, X_field, layouttype)
        x_turbines = hcat(x_turbines, next_turbine)
        no_of_turbines = no_of_turbines - 1
    end

    expected_revenue = turbine_approximate_profits(x_turbines, gpla_wf, tlparams)
    return x_turbines, expected_revenue
end

function get_next_turbine_location(gpla_wf, X_field, layouttype::GreedyTurbineLayout)

    μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
    σ = sqrt.(σ²)
    N = max(1, length(gpla_wf.y))

    z_value = 1.645    # chosen: 90 percent confidence interval
    LCB = μ - z_value / sqrt(N) * σ
    LCB_vec = vec(LCB)
    if isempty(LCB_vec)
        error("There isn't enough space on the field for the specified no_of_turbines.")
    else
        best_val = argmax(LCB_vec)
    end
    
    next_turbine = X_field[:, best_val]

    return next_turbine
end