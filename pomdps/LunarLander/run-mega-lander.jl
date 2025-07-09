using POMDPs
using POMDPTools
using POMCPOW
using BOMCP

using CPUTime
using Dates
using Distributions
using LinearAlgebra
using Parameters
using ParticleFilters
using Random
using Statistics
using StatsBase

include("po_lunar.jl")
include("pa_pomcpow.jl")
include("action_select.jl")

function action_values(p::LunarLander, b::ParticleCollection{Vector{Float64}})
    # todo
    # 对LanderActionSpace的处理
end

formatted_string = Dates.format(now(), "yymmddHHMM")
filename = string("lander_mega_pomcpow_", formatted_string, ".txt")
if !isfile(filename)
    touch(filename)
end
io = open(filename, "w")

Random.seed!(42)

for k = 1:1
    adr_sum = 0.0
    ep = 1

    total_time = 0.0
    sim_times = Float64[]

    while ep <= 1000
        try
            sim_start = time()

            pomdp = LunarLander()

            b0 = initialstate(pomdp)

            updater = BootstrapFilter(pomdp, 1000)

            action_selector = ActionSelector{Vector{Float64}}(action_values, collect(0.0:0.1:2.0))
            solver = PAPOMCPOWSolver(
                tree_queries=1000,
                next_action=action_selector,
                belief_updater=updater,
                max_depth=20,
                alpha_observation=0.5,
                k_observation=10.0,
                alpha_action=0.5,
                k_action=10.0
            )

            planner = solve(solver, pomdp)

            hr = HistoryRecorder(max_steps=100)
            hist = POMDPs.simulate(hr, pomdp, planner, updater, b0)
            adr = discounted_reward(hist)
            adr_sum += adr
            avg_adr = adr_sum / ep

            sim_end = time()
            sim_time = sim_end - sim_start
            push!(sim_times, sim_time)
            total_time += sim_time
            avg_time = total_time / ep

            println("""
                Simulation $(ep)
                    MEGA-POMCPOW:
                        ADR: $(adr)
                        Time: $(round(sim_time, digits=4)) s
                Cumulative Statistics (for $(ep) simulation)
                    MEGA-POMCPOW:
                        AVG_ADR: $(avg_adr)
                        AVG_Time: $(round(avg_time, digits=4)) s
            """)
            write(
                io,
                """
                Simulation $(ep)
                    MEGA-POMCPOW:
                        ADR: $(adr)
                        Time: $(round(sim_time, digits=4)) s
                Cumulative Statistics (for $(ep) simulation)
                    MEGA-POMCPOW:
                        AVG_ADR: $(avg_adr)
                        AVG_Time: $(round(avg_time, digits=4)) s
                """
            )
            ep += 1
        catch ex
            println("Error in simulation: ", ex)
        end
    end
end

close(io)
