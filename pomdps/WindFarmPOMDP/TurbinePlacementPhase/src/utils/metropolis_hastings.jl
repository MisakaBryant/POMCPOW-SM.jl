function mcmc(init_state, obj_func, no_of_iterations, X_field, tlparams; tune_factor = 1)
""" Optimizes state using Metropolis–Hastings MCMC (Markov Chain Monte Carlo), as described in Tzanos et al. """

    prev_state = copy(init_state)
    best_state = copy(init_state)
    
    while no_of_iterations > 0
        
        state_dist, state_dist_vals = fit_distribution(prev_state, X_field)
        next_state, next_state_prob = sample_layout(state_dist, state_dist_vals, X_field, tlparams)
        prev_state_prob = recover_prev_state_prob(prev_state, next_state, X_field)

        J_prev = obj_func(prev_state)
        J_next = obj_func(next_state)
        J_ratio = (J_next / J_prev)^tune_factor

        ρ = min(1, (prev_state_prob / next_state_prob * J_ratio))
        
        obj_func(best_state) ≤ obj_func(next_state) ? best_state = copy(next_state) : nothing
        rand() ≤ ρ ? prev_state = next_state : nothing

        no_of_iterations = no_of_iterations - 1
    end
    return best_state
end

function fit_distribution(x_turbines, X_field; num_of_neighbors = 15)
    kdtree = NearestNeighbors.KDTree(X_field)
    knn_results = knn.(Ref(kdtree), eachcol(x_turbines), Ref(num_of_neighbors))

    state_dist_vals = nn = getindex.(knn_results, Ref(1))

    # pdfs = [exp.((item[2]./ℓ).^2) for item in knn_results]
    pdfs = get_pdfs.(knn_results)
    pdfs_nrmz = prob_normalize.(pdfs)

    state_dist = Distributions.Categorical.(pdfs_nrmz)
    return state_dist, state_dist_vals
end

get_pdfs(knn_results_single_col; ℓ = 220 * 5) = exp.((knn_results_single_col[2]./ℓ).^2)

function sample_layout(state_dist, state_dist_vals, X_field, tlparams; n_tries = 50)
    spl = reshape(Float64[], 3, 0)
    spl_prob = 0.0

    while n_tries > 0
        # @show n_tries
        idxs = rand.(state_dist)
        spl_prob = prod(getindex.(probs.(state_dist), idxs))
        spl = X_field[:, getindex.(state_dist_vals, idxs)]
        if !is_solution_separated(spl, tlparams)
            # @show is_solution_separated_Int(spl, tlparams)
            return spl, spl_prob
        end
        n_tries = n_tries - 1
    end
    return spl, spl_prob
end

function recover_prev_state_prob(prev_state, next_state, X_field; num_of_neighbors = 15)
    kdtree = NearestNeighbors.KDTree(X_field)
    knn_results = knn.(Ref(kdtree), eachcol(next_state), Ref(num_of_neighbors))
    nn = getindex.(knn_results, Ref(1))

    prev_state_idxs = knn.(Ref(kdtree), eachcol(prev_state), Ref(1))
    prev_state_idxs = getindex.(getindex.(prev_state_idxs, Ref(1)), Ref(1))

    # pdfs = [exp.((item[2]./ℓ).^2) for item in knn_results]
    pdfs = get_pdfs.(knn_results)
    pdfs = sum.(pdfs)

    return prod(inv.(pdfs))
end