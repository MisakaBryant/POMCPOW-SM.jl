global HEATMAP_μ_CLIM = (0.0, 15.0)
global HEATMAP_σ_CLIM = (0.0, 3.0)

transform_FieldCoord_to_PlotCoord(x, wfparams) = x[1:2] / wfparams.grid_dist + [1,1]    # [1,1] is added because locations are 0-based indexed.

function save_rewards_to_disk(script_id::Symbol, rewards_history::AbstractArray, savename::String)
    println("### Total Rewards: $(sum(rewards_history)) ###")
    open(savename, "a") do io
        writedlm(io, ["Script: ", string(script_id, ".jl"), ""])
        writedlm(io, ["History: ", rewards_history..., ""])
        writedlm(io, ["Total: ", sum(rewards_history)])
    end
    return nothing
end

function plot_WindFarmPOMDP_policy(script_id::Symbol, wfparams::WindFieldBeliefParams, actions_history::AbstractArray, rewards_history::AbstractArray, b0::WindFarmBelief; savetype = :png)
    println("### Creating Policy Plots ###")
    savetype = String(savetype)
    nx, ny = wfparams.nx, wfparams.ny
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    actions_history = CartIndices_to_Vector.(actions_history)
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    gpla_wf = b0.gpla_wf
    
    for h in wfparams.altitudes

        # Extract the actions `a` taken that were at altitude `h`.
        a_in_h = reshape(Float64[],2,0)
        for a in actions_history
            if a[end]==h
                a_in_h = hcat(a_in_h, a[1:2] / wfparams.grid_dist + [1,1])    # [1,1] is added because locations are 0-based indexed.
            end
        end
        
        X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
        p = Plots.heatmap(reshape(Y_field, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Wind Farm Sensor Locations Chosen, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p, "./$dir/Plot_$h.$savetype")
    
        # Plots of initial belief below.
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
        σ = sqrt.(σ²)

        p2 = Plots.heatmap(reshape(σ, (nx,ny)), clim=HEATMAP_σ_CLIM, title="Wind Farm Initial Belief Variance, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p2, "./$dir/Plot2_$h.$savetype")

        p3 = Plots.heatmap(reshape(μ, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Wind Farm Initial Belief Mean, h = $(h)m")
        Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
        Plots.savefig(p3, "./$dir/Plot3_$h.$savetype")

    end
    save_rewards_to_disk(script_id, rewards_history, "./$dir/rewards.txt")
    
    println("### Policy Plots Saved to $dir ###")
    return nothing
end

function plot_WindFarmPOMDP_belief_history(wfparams::WindFieldBeliefParams, actions_history::AbstractArray, belief_history::AbstractArray, b0::WindFarmBelief; savetype = :png)
    println("### Creating Policy Plots ###")
    nx, ny = wfparams.nx, wfparams.ny
    Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
    actions_history = CartIndices_to_Vector.(actions_history)
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    for (idx,b) in enumerate(vcat(b0, belief_history))
        t = idx - 1     # timestep
        for h in wfparams.altitudes

            # Extract the actions `a` taken that were at altitude `h`.
            a_in_h = reshape(Float64[],2,0)

            if idx != 1
                for a in actions_history[1:t]
                    if a[end]==h
                        a_in_h = hcat(a_in_h, a[1:2] / wfparams.grid_dist + [1,1])    # [1,1] is added because locations are 0-based indexed.
                    end
                end
            end

            # Extract field at `h`.
            X_field, Y_field = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
            
            # Plots of initial belief below.
            gpla_wf = b.gpla_wf
            μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
            σ = sqrt.(σ²)

            p2 = Plots.heatmap(reshape(σ, (nx,ny)), clim=HEATMAP_σ_CLIM, title="Std of Belief at t=$(t), h = $(h)m")
            Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
            Plots.savefig(p2, "./$dir/PlotVar_t$(t)_h$(h).$savetype")

            p3 = Plots.heatmap(reshape(μ, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Mean of Belief at t=$(t), h = $(h)m")
            Plots.scatter!(a_in_h[2,:], a_in_h[1,:], legend=false, color=:white)  # Notice that the row and col of `a_in_h` is reversed.
            Plots.savefig(p3, "./$dir/PlotMean_t$(t)_h$(h).$savetype")

        end
    end

    println("### Policy Plots Saved to $dir ###")
    return nothing
end


# Parses `A` into the correct AbstractArray subtype
plot_WindFarmPOMDP_TPP_history(wfparams::WindFieldBeliefParams, A::AbstractArray, belief_history::AbstractArray, b0::WindFarmBelief; sensor_color = :white, turbine_color = :lightgreen, savetype = :png) = plot_WindFarmPOMDP_TPP_history(wfparams, [item for item in A], belief_history, b0; sensor_color = :white, turbine_color = :lightgreen, savetype = :png)

# For Sequential solver solutions
function plot_WindFarmPOMDP_TPP_history(wfparams::WindFieldBeliefParams, actions_history::AbstractArray{CartesianIndex{N},T} where {N,T}, belief_history::AbstractArray, b0::WindFarmBelief; 
                                        sensor_color = :white, turbine_color = :lightgreen, savetype = :png)
    println("### Creating Turbine Placement Phase Plots ###")
    nx, ny = wfparams.nx, wfparams.ny
    actions_history = CartIndices_to_Vector.(actions_history)
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    for (idx,b) in enumerate(vcat(b0, belief_history))
        t = idx - 1     # timestep
        h = wfparams.altitudes[end]

        # Extract the actions `a` taken that were at altitude `h`.
        a_in_t = reshape(Float64[],2,0)

        if idx != 1
            for a in actions_history[1:t]
                a_in_t = hcat(a_in_t, a[1:2] / wfparams.grid_dist + [1,1])    # [1,1] is added because locations are 0-based indexed.
            end
        end

        # Extract field at `h`.
        Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
        X_field, _ = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
        
        # Plots of initial belief below.
        gpla_wf = b.gpla_wf
        μ, σ² = GaussianProcesses.predict_f(gpla_wf, X_field)
        σ = sqrt.(σ²)

        # Get LCB of mean.
        N = max(1, length(gpla_wf.y))
        z_value = 1.645    # chosen: 90 percent confidence interval
        LCB = μ - z_value / sqrt(N) * σ
        LCB_vec = vec(LCB)
        @show extrema(LCB_vec)


        # Get the turbine layout for this belief
        x_turbines, _ = get_turbine_layout(gpla_wf, tlparams, wfparams, tlparams.layouttype)
        x_turbines = hcat(transform_FieldCoord_to_PlotCoord.(eachcol(x_turbines), Ref(wfparams))...)

        p2 = Plots.heatmap(reshape(σ, (nx,ny)), clim=HEATMAP_σ_CLIM, title="Std of Belief at t=$(t), h = $(h)m")
        Plots.scatter!(x_turbines[2,:], x_turbines[1,:], m=:square, legend=false, color=turbine_color)     # Notice that the row and col of `a_in_t` is reversed.
        Plots.scatter!(a_in_t[2,:], a_in_t[1,:], legend=false, color=sensor_color)                         # Notice that the row and col of `a_in_t` is reversed.
        Plots.savefig(p2, "./$dir/PlotVar_t$(t)_h$(h).$savetype")

        p3 = Plots.heatmap(reshape(μ, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Mean of Belief at t=$(t), h = $(h)m")
        Plots.scatter!(x_turbines[2,:], x_turbines[1,:], m=:square, legend=false, color=turbine_color)     # Notice that the row and col of `a_in_t` is reversed.
        Plots.scatter!(a_in_t[2,:], a_in_t[1,:], legend=false, color=sensor_color)                         # Notice that the row and col of `a_in_t` is reversed.
        Plots.savefig(p3, "./$dir/PlotMean_t$(t)_h$(h).$savetype")

        p4 = Plots.heatmap(reshape(LCB_vec, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Mean LCB of Belief at t=$(t), h = $(h)m")
        Plots.scatter!(x_turbines[2,:], x_turbines[1,:], m=:square, legend=false, color=turbine_color)     # Notice that the row and col of `a_in_t` is reversed.
        Plots.scatter!(a_in_t[2,:], a_in_t[1,:], legend=false, color=sensor_color)                         # Notice that the row and col of `a_in_t` is reversed.
        Plots.savefig(p4, "./$dir/PlotLCB_t$(t)_h$(h).$savetype")

    end

    println("### Plots Saved to $dir ###")
    return nothing
end

# For Non-Sequential solver solutions
function plot_WindFarmPOMDP_TPP_history(wfparams::WindFieldBeliefParams, soln::AbstractArray{Float64,N} where N, b0::WindFarmBelief, s0::WindFarmState; 
                                        sensor_color = :white, turbine_color = :lightgreen, savetype = :png)
    println("### Creating Turbine Placement Phase Plots ###")
    nx, ny = wfparams.nx, wfparams.ny
    actions_history = collect(eachcol(soln))
    
    !isdir("Figures") ? mkdir("Figures") : nothing
    dir = string("Figures/", Dates.now())
    mkdir(dir)

    # Get ground truth
    x_obs_full = s0.x_obs_full
    y_obs_full = s0.y_obs_full
    gpla_wf_full = get_GPLA_for_gen(x_obs_full, y_obs_full, wfparams)       # Ground truth.

    # Get latest belief
    x_obs = x_sensors = soln
    y_obs = rand(gpla_wf_full, x_obs)
    b1_gpla_wf = get_GPLA_for_gen(x_obs, y_obs, wfparams)                      # Latest belief based on previous observations.


    for (idx, b_gpla_wf) in enumerate(vcat(b0.gpla_wf, b1_gpla_wf))
        t = idx - 1     # timestep
        h = wfparams.altitudes[end]

        # Extract the actions `a` taken that were at altitude `h`.
        a_in_t = reshape(Float64[],2,0)

        if idx == 1
            a_in_t = reshape(Float64[],2,0)
        else
            a_in_t = hcat(transform_FieldCoord_to_PlotCoord.(eachcol(soln), Ref(wfparams))...)
        end

        # Extract field at `h`.
        Map = get_3D_data(wfparams.farm; altitudes=wfparams.altitudes)
        X_field, _ = get_dataset(Map, [h], wfparams.grid_dist, wfparams.grid_dist, 1, wfparams.nx, 1, wfparams.ny)
        
        # Plots of initial belief below.
        μ, σ² = GaussianProcesses.predict_f(b_gpla_wf, X_field)
        σ = sqrt.(σ²)

        # Get LCB of mean.
        N = max(1, length(b_gpla_wf.y))
        z_value = 1.645    # chosen: 90 percent confidence interval
        LCB = μ - z_value / sqrt(N) * σ
        LCB_vec = vec(LCB)


        # Get the turbine layout for this belief
        x_turbines, _ = get_turbine_layout(b_gpla_wf, tlparams, wfparams, tlparams.layouttype)
        x_turbines = hcat(transform_FieldCoord_to_PlotCoord.(eachcol(x_turbines), Ref(wfparams))...)

        p2 = Plots.heatmap(reshape(σ, (nx,ny)), clim=HEATMAP_σ_CLIM, title="Std of Belief at t=$(t), h = $(h)m")
        Plots.scatter!(x_turbines[2,:], x_turbines[1,:], m=:square, legend=false, color=turbine_color)     # Notice that the row and col of `a_in_t` is reversed.
        Plots.scatter!(a_in_t[2,:], a_in_t[1,:], legend=false, color=sensor_color)                         # Notice that the row and col of `a_in_t` is reversed.
        Plots.savefig(p2, "./$dir/PlotVar_t$(t)_h$(h).$savetype")

        p3 = Plots.heatmap(reshape(μ, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Mean of Belief at t=$(t), h = $(h)m")
        Plots.scatter!(x_turbines[2,:], x_turbines[1,:], m=:square, legend=false, color=turbine_color)     # Notice that the row and col of `a_in_t` is reversed.
        Plots.scatter!(a_in_t[2,:], a_in_t[1,:], legend=false, color=sensor_color)                         # Notice that the row and col of `a_in_t` is reversed.
        Plots.savefig(p3, "./$dir/PlotMean_t$(t)_h$(h).$savetype")

        p4 = Plots.heatmap(reshape(LCB_vec, (nx,ny)), clim=HEATMAP_μ_CLIM, title="Mean LCB of Belief at t=$(t), h = $(h)m")
        Plots.scatter!(x_turbines[2,:], x_turbines[1,:], m=:square, legend=false, color=turbine_color)     # Notice that the row and col of `a_in_t` is reversed.
        Plots.scatter!(a_in_t[2,:], a_in_t[1,:], legend=false, color=sensor_color)                         # Notice that the row and col of `a_in_t` is reversed.
        Plots.savefig(p4, "./$dir/PlotLCB_t$(t)_h$(h).$savetype")

    end

    println("### Plots Saved to $dir ###")
    return nothing
end


""" Global versions below """
save_rewards_to_disk() = save_rewards_to_disk(script_id, rewards_history, savename)
plot_WindFarmPOMDP_policy() = plot_WindFarmPOMDP_policy(script_id, wfparams, actions_history, rewards_history, b0)
plot_WindFarmPOMDP_belief_history() = plot_WindFarmPOMDP_belief_history(wfparams, actions_history, belief_history, b0)
plot_WindFarmPOMDP_TPP_history() = plot_WindFarmPOMDP_TPP_history(wfparams, actions_history, belief_history, b0)
plot_WindFarmPOMDP_TPP_history_non_seq() = plot_WindFarmPOMDP_TPP_history(wfparams, soln, b0, s0) 