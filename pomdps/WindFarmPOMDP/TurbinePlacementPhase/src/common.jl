"""
    Common functions used in the Turbine Placement Phase.
"""

abstract type TurbineLayoutType end

@with_kw struct TurbineLayoutParams
    # Layout params
    layouttype::TurbineLayoutType = GreedyTurbineLayout()
    grid_dist::Int = 220                                        # [meters]
    altitudes::Vector = [100, 200]                              # [meters]

    # Wind unit-direction
    wind_direction::Vector = [1,1,0]

    # Sensor tower specs
    mast_cost::Float64 = 2.0e3                                  # [USD/meter]

    # Turbine specs
    no_of_turbines::Int = 10
    turbine_cost::Float64 = 4.0e6                               # [USD/turbine]
    turbine_diameter::Int = 120                                 # [meters]
    turbine_max_power::Int = 2500                               # [kW]
    turbine_cut_in_speed::Float64 = 3.0                         # [m/s]
    turbine_rated_speed::Float64 = 11.0                         # [m/s]
    turbine_power_curve::Polynomial = create_power_curve()      # [kW, given windspeed]
end

function create_power_curve(poly_degree::Int = 3, data_dir = "../../TurbinePlacementPhase/data/ge_turbine_data.csv")
    power_data = DelimitedFiles.readdlm(data_dir, Float64, comments=true, comment_char='#')
    speed_ms, power_kW = eachcol(power_data)
    curve_fit = Polynomials.fit(speed_ms, power_kW, poly_degree)
    return curve_fit
    # scatter(speed_ms, power_kW, label="Data")
    # plot!(curve_fit, extrema(speed_ms)..., label="Fit")
end

function TurbineLayoutParams(layoutfinder::Symbol)
    layout_dict = Dict(
        :greedy         => GreedyTurbineLayout,
        :genetic        => GeneticTurbineLayout,
        :mcmc           => MCMCTurbineLayout        
    )

    layouttype = layout_dict[layoutfinder]()
    return TurbineLayoutParams(layouttype=layouttype)
end

function get_seperation_heuristic(xi, xj, ui, uj, turbine_diameter)
""" Find the minimum required distance for two coordinates to be out of each others' wake seperation regions. """ 
    
    # Use ηmin = 3D and ηmax = 10D, as commonly done.
    ηmin, ηmax = [3, 10] .* turbine_diameter

    if view(xi, 1:2) == view(xj, 1:2) 
        return ηmax
    else
        vij = 0.5 * (ui + uj)
        dij = xi - xj
        ηij = ηmin + (ηmax - ηmin) * norm(dot_product(dij, vij)) / norm(dij) / norm(vij)
        return ηij
    end
end

function is_solution_separated(x_turbines, tlparams)
""" Check if any point in `x_turbines` are inside another's seperation region. """ 
    ui = uj = tlparams.wind_direction
    dia = tlparams.turbine_diameter
    idx_set = collect(1:size(x_turbines, 2))

    if isempty(x_turbines)
        return false
    else
        for (idx, xi) in enumerate(eachcol(x_turbines))
            idx_complement = setdiff(idx_set, collect(1:idx))
            x_turbines_complement = x_turbines[:, idx_complement]
            η = get_seperation_heuristic.(Ref(xi), eachcol(x_turbines_complement), Ref(ui), Ref(uj), Ref(dia))
            d = euclidean_dist.(Ref(xi), eachcol(x_turbines_complement))
            all(η .< d) ? nothing : return true
        end
    end

    return false    # return false if solution is NOT separated.
end

function is_solution_separated_Int(x_turbines, tlparams)
""" Check if any point in `x_turbines` are inside another's seperation region. Returns number of placements under seperation. """ 
    ui = uj = tlparams.wind_direction
    dia = tlparams.turbine_diameter
    idx_set = collect(1:size(x_turbines, 2))

    if isempty(x_turbines)
        return 0
    else
        sep = 0
        for (idx, xi) in enumerate(eachcol(x_turbines))
            idx_complement = setdiff(idx_set, collect(1:idx))
            x_turbines_complement = x_turbines[:, idx_complement]
            η = get_seperation_heuristic.(Ref(xi), eachcol(x_turbines_complement), Ref(ui), Ref(uj), Ref(dia))
            d = euclidean_dist.(Ref(xi), eachcol(x_turbines_complement))
            sep = sep + sum(η .≥ d) 
        end
        return sep
    end
end
    
function remove_seperated_locations(X_field, x_turbines, tlparams)
""" Remove the locations in `X_field` that are effected by the seperation region of any point in `x_turbines`.
    This reduces the turbine placement action space. """

    ui = uj = tlparams.wind_direction
    dia = tlparams.turbine_diameter

    if isempty(x_turbines)
        nothing
    else
        for xi in eachcol(x_turbines)
            η = get_seperation_heuristic.(Ref(xi), eachcol(X_field), Ref(ui), Ref(uj), Ref(dia))
            d = euclidean_dist.(Ref(xi), eachcol(X_field))
            X_field = X_field[:, η .< d]
        end
        X_field = remove_same_coordinate_locations(X_field, x_turbines)
    end

    return X_field
end

function find_same_coordinate_locations(xi, X_field)
""" Find the locations in `X_field` that are in the same coordinate as `xi`. """
    result = []

    for (idx, xj) in enumerate(eachcol(X_field))
        if view(xi, 1:2) == view(xj, 1:2) 
            push!(result, idx)
        end
    end
    return result
end

function remove_same_coordinate_locations(X_field, x_turbines)
""" Remove the locations in `X_field` that are in the same coordinate with any of the locations in `x_turbines`. """
    locs = find_same_coordinate_locations.(eachcol(x_turbines), Ref(X_field))
    locs_complement = setdiff(collect(1:size(X_field, 2)), vcat(locs...))
    return X_field[:, locs_complement]
end

function get_tower_cost(a)
""" Get the cost of a single mast (sensor) tower. """
    height = a[end]
    return height * tlparams.mast_cost
end

function get_turbine_cost(a)
""" Get the cost of a single turbine. """
    return tlparams.turbine_cost
end

function turbine_action_space(tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams)
""" Get the entire action space for possible turbine placements. """
    altitudes = tlparams.altitudes
    grid_dist = tlparams.grid_dist

    x_ranges = 0 : grid_dist : grid_dist * wfparams.nx-1
    y_ranges = 0 : grid_dist : grid_dist * wfparams.ny-1
    all_actions = [CartesianIndex(x,y,z) for x in x_ranges for y in y_ranges for z in altitudes]

    return vec(collect(all_actions))
end

function get_random_init_solution(X_field, no_of_turbines, tlparams)
""" Returns a random initial valid turbine placement layout. """ 
    kdtree = NearestNeighbors.KDTree(X_field)
    x_turbines = reshape(Float64[], 3, 0)

    while no_of_turbines > 0
        X_field = remove_seperated_locations(X_field, x_turbines, tlparams)

        next_loc = rand(1:size(X_field, 2))
        next_turbine = X_field[:, next_loc]

        x_turbines = hcat(x_turbines, next_turbine)
        no_of_turbines = no_of_turbines - 1
    end

    knn_results = knn.(Ref(kdtree), eachcol(x_turbines), Ref(1))
    nn = getindex.(knn_results, Ref(1))
    locs = vcat(nn...)

    return x_turbines, locs
end

function get_average_turbine_distances(x_turbines)
""" Calculate the average spacing [meters] between turbines. """
    dists = Float64[]
    idx_set = collect(1:size(x_turbines, 2))

    while length(idx_set) > 1
        # @show idx_set
        idx = pop!(idx_set)
        xi = x_turbines[:, idx]
        d = euclidean_dist.(Ref(xi), eachcol(x_turbines[:,idx_set]))
        push!(dists, minimum(d))
    end   
    return mean(dists)
end

function turbine_approximate_profits(x_turbines::AbstractMatrix, gpla_wf, tlparams; penalty_cost = 2.0e6)
""" Approximate sum of profits of individual turbines. """
    
    # From belief
    μ, σ² = GaussianProcesses.predict_f(gpla_wf, x_turbines)
    σ = sqrt.(σ²)
    N_samples = max(1, length(gpla_wf.y))
    
    costs = get_turbine_cost.(eachcol(x_turbines))
    s_avg = get_average_turbine_distances(x_turbines) / tlparams.turbine_diameter
    profits = get_turbine_profit.(μ, σ, costs, Ref(s_avg), Ref(N_samples), Ref(tlparams))

    result = sum(profits) - penalty_cost * sum(is_solution_separated_Int(x_turbines, tlparams))    # Penalty added for solutions within seperated regions.
    return Int(round(result))
end

function get_layout_profit(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::TurbineLayoutType)
""" Calculate approximate profit of a turbine layout. """
    # Cost of sensor tower placements
    x_sensors = sp.x_acts
    cost_masts = get_tower_cost.(eachcol(x_sensors))

    # Profit of turbine placements
    x_turbines, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, layouttype)
    expected_turbine_profits = turbine_approximate_profits(x_turbines, gpla_wf, tlparams)

    # total_profit = sum(expected_turbine_profits) - sum(cost_masts)
    # return total_profit
    return sum(expected_turbine_profits)
end

turbine_approximate_profits(locs::AbstractVector, X_field, gpla_wf, tlparams) = turbine_approximate_profits(X_field[:, locs], gpla_wf, tlparams)
get_layout_profit(sp::WindFarmState, gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams) = get_layout_profit(sp, gpla_wf, tlparams, wfparams, tlparams.layouttype)

function turbine_ground_profits(x_turbines::AbstractMatrix, x_obs_full, y_obs_full, gpla_wf, tlparams; penalty_cost = 2.0e6)
""" Approximate sum of profits of individual turbines. """
    
    # From ground truth
    kdtree_full = NearestNeighbors.KDTree(x_obs_full)
    y_obs_idx = knn.(Ref(kdtree_full), eachcol(x_turbines), Ref(1))
    y_obs_idx = getindex.(y_obs_idx, Ref(1))
    ui_vals = y_obs_full[vcat(y_obs_idx...)]

    # From belief
    _, σ² = GaussianProcesses.predict_f(gpla_wf, x_turbines)
    σi_vals = sqrt.(σ²)
    N_samples = max(1, length(gpla_wf.y))
    
    costs = get_turbine_cost.(eachcol(x_turbines))
    s_avg = get_average_turbine_distances(x_turbines) / tlparams.turbine_diameter
    profits = get_turbine_profit.(ui_vals, σi_vals, costs, Ref(s_avg), Ref(N_samples), Ref(tlparams))   #L305.

    result = sum(profits) - penalty_cost * sum(is_solution_separated_Int(x_turbines, tlparams))    # Penalty added for solutions within seperated regions.
    return Int(round(result))
end

function get_ground_truth_profit(states_history::AbstractArray, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::TurbineLayoutType)
""" Calculate approximate profit of a turbine layout using ground truth, from the final state. Called after sequential solvers. """

    # Re-seed ascertain the randomness among all scripts.
    Random.seed!(wfparams.noise_seed)

    # Cost of sensor tower placements
    s_final = states_history[end]
    x_sensors = s_final.x_acts
    cost_masts = get_tower_cost.(eachcol(x_sensors))

    # Get belief and ground truth
    x_obs_full = s_final.x_obs_full
    y_obs_full = s_final.y_obs_full
    gpla_wf = get_GPLA_for_gen(s_final.x_obs, s_final.y_obs, wfparams)                      # Latest belief based on previous observations.
    gpla_wf_full = get_GPLA_for_gen(x_obs_full, y_obs_full, wfparams)       # Ground truth.

    # Profit of turbine placements
    x_turbines, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, layouttype)
    expected_turbine_profits = turbine_ground_profits(x_turbines, x_obs_full, y_obs_full, gpla_wf, tlparams)  #L237.

    # total_profit = sum(expected_turbine_profits) - sum(cost_masts)
    # return total_profit
    return sum(expected_turbine_profits)
end

function get_ground_truth_profit(s0::WindFarmState, x_sensors::AbstractArray, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::TurbineLayoutType)
""" Calculate approximate profit of a turbine layout using ground truth, from initial state and solution found. Called after non-sequential solvers. """

    # Re-seed ascertain the randomness among all scripts.
    Random.seed!(wfparams.noise_seed)

    # Cost of sensor tower placements
    cost_masts = get_tower_cost.(eachcol(x_sensors))
    
    # Get ground truth
    x_obs_full = s0.x_obs_full
    y_obs_full = s0.y_obs_full
    gpla_wf_full = get_GPLA_for_gen(x_obs_full, y_obs_full, wfparams)       # Ground truth.

    # Get latest belief
    x_obs = x_sensors
    y_obs = rand(gpla_wf_full, x_obs)
    gpla_wf = get_GPLA_for_gen(x_obs, y_obs, wfparams)                      # Latest belief based on previous observations.

    # Profit of turbine placements
    x_turbines, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, layouttype)
    expected_turbine_profits = turbine_ground_profits(x_turbines, x_obs_full, y_obs_full, gpla_wf, tlparams)

    # total_profit = sum(expected_turbine_profits) - sum(cost_masts)
    # return total_profit
    return sum(expected_turbine_profits)
end

get_ground_truth_profit(states_history::AbstractArray, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams) = get_ground_truth_profit(states_history, tlparams, wfparams, tlparams.layouttype)
get_ground_truth_profit(s0::WindFarmState, x_sensors::AbstractArray, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams) = get_ground_truth_profit(s0, x_sensors, tlparams, wfparams, tlparams.layouttype)

function get_turbine_profit(ui, σi, turbine_cost, s_avg, N_samples, tlparams)
""" Follows the heuristic as described in Stevens et al. """
    # θ = 1.0e-3
    # β = 5.0e-3
    # γ = 4

    # P∞ = get_turbine_power_output(ui, tlparams)
    # P₁ = tlparams.turbine_max_power

    # # Calculate the ratio between Revenue and Costs, for a single turbine.
    # profit_star = (P∞ / P₁) * γ - (1 + β * s_avg + θ * s_avg^2)

    # return profit_star * turbine_cost
    
    # return ui^3

    z_value = 1.645    # chosen: 90 percent confidence interval
    LCB = ui - z_value / sqrt(N_samples) * σi
    return LCB^3
end

function get_turbine_power_output(ui, tlparams)
""" Keeps the power output within reasonable limits. """
    ui_clamped = clamp(ui, tlparams.turbine_cut_in_speed, tlparams.turbine_rated_speed)
    return tlparams.turbine_power_curve(ui_clamped)
end