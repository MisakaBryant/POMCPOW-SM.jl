"""
    Genetic method to heuristically determine optimal turbine layout.
"""

@with_kw struct GeneticTurbineLayout <: TurbineLayoutType
    no_of_iterations = 600
    populationSize = 6000
    crossoverRate = 0.8
    mutationRate = 0.05
end


function get_turbine_layout(gpla_wf::GPLA, tlparams::TurbineLayoutParams, wfparams::WindFieldBeliefParams, layouttype::GeneticTurbineLayout)
    
    greedy_layout, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, GreedyTurbineLayout())
    no_of_turbines = tlparams.no_of_turbines
    X_field = CartIndices_to_Array(turbine_action_space(tlparams, wfparams))
    size_X_field = size(X_field, 2)
    
    function init_locs()
        # _, loc = get_random_init_solution(X_field, no_of_turbines, tlparams)

        kdtree = NearestNeighbors.KDTree(X_field)
        knn_results = knn.(Ref(kdtree), eachcol(greedy_layout), Ref(10))
        nn = getindex.(knn_results, Ref(1))

        return rand.(nn)
    end

    function cons(locs, X_field)
        x_turbines = X_field[:, locs]
        return [ sum(is_solution_separated_Int(x_turbines, tlparams)) ]
    end
    
    function mutation_func!(x)
        x[:] = rand(1:size_X_field, no_of_turbines)
    end


    lx = fill(1, no_of_turbines)
    ux = fill(size_X_field, no_of_turbines)
    tc = fill(Int, no_of_turbines)

    lc, uc = [0.0], [0.0]

    cb = Evolutionary.ConstraintBounds(lx,ux,lc,uc)
    constraints = MixedTypePenaltyConstraints(PenaltyConstraints([1e3], cb, x -> cons(x, X_field)), tc)

    opts = Evolutionary.Options(iterations = layouttype.no_of_iterations, abstol = 1e-5)

    mthd = GA(populationSize = layouttype.populationSize,
              crossoverRate = layouttype.crossoverRate,
              mutationRate = layouttype.mutationRate,
              selection = Evolutionary.sus,
              crossover = Evolutionary.uniform,
              mutation = mutation_func!
    )

    obj_func = locs -> - turbine_approximate_profits(locs, X_field, gpla_wf, tlparams)      # Note the negative sign, since GA is a minimizer.

    GA_result = Evolutionary.optimize(obj_func,
                                      constraints,
                                      init_locs,
                                      mthd,
                                      opts
    )
    
    x_turbines = X_field[:, GA_result.minimizer]
    expected_revenue = - GA_result.minimum                                                  # Note the negative sign, since GA is a minimizer.
    return x_turbines, expected_revenue
end