# Parse ARGS
solvermethod = :random
layoutfinders = [:greedy, :genetic, :mcmc]
savename = isempty(ARGS) ? "compare_tpp_solvers.csv" : ARGS[end]

include("../../SensorPlacementPhase/src/SensorPP.jl")
include("../../TurbinePlacementPhase/src/TurbinePP.jl")


# Wind Field Belief Parameters
wfparams = WindFieldBeliefParams()

# Turbine Layout Heuristic Parameters
tlparams = TurbineLayoutParams()

# Construct POMDP
no_of_sensors = 5
delta = 220 * 4
pomdp = WindFarmPOMDP(wfparams.nx, wfparams.ny, wfparams.grid_dist, wfparams.altitudes, no_of_sensors, delta)

# Get initial belief distribution and initial state
b0 = initialize_belief_lookup(wfparams)
s0 = initialize_state(b0, wfparams)

# Define Solver
solver = extract_solver_method(pomdp, solvermethod)

# Compute the optimal sensor placements
soln = get_solution(s0, pomdp, tlparams, wfparams, solver)

layout_dict = Dict(
    :greedy         => GreedyTurbineLayout,
    :genetic        => GeneticTurbineLayout,
    :mcmc           => MCMCTurbineLayout        
)

RR_to_save = []

for layoutfinder in layoutfinders
    
    # Show utility of solution
    time_taken = @elapsed begin
        RR = get_ground_truth_profit(s0, soln, tlparams, wfparams, layout_dict[layoutfinder]())
    end
    
    # Save results
    @show layoutfinder, round(time_taken; digits=3)
    push!(RR_to_save, layoutfinder)
    push!(RR_to_save, RR)
    
end

writedlm_append(savename, hcat(solvermethod, layoutfinder, round(time_taken; digits=2), vec(RR_to_save)))