# Parses the data provided in ../data/DWA/

struct data_2D
    x:: Array{Float64}
    y:: Array{Float64}
    value:: Array{Float64}

    function data_2D(datapath, filename)
        data_read = DelimitedFiles.readdlm(datapath*filename, ' ', Float64)
        
        x = data_read[:,1]
        y = data_read[:,2]
        value = data_read[:,3]

        new(x,y,value)
    end
end

function get_3D_data(farm; altitudes = [10, 50, 100, 150, 200], loc_parent = "pomdps/WindGP/data/GWA")
    
    filename_t(h) = "custom_wind-speed_$(h)m.xyz"
    datapath_t(loc) = "$(loc_parent)/$(loc)/"
    datapath = datapath_t(farm)
    
    data_3D = Dict{Int, Array{Float64,2}}()

    for h in altitudes
        filename = filename_t(h)
        D = data_2D(datapath, filename)
        h_data = reshape(D.value, length(unique(D.x)), length(unique(D.y)))
        
        push!(data_3D, h=>h_data)
    end

    return data_3D
end

function get_dataset(Map, altitudes, grid_dist_mid, grid_dist, nx_start, nx_end, ny_start, ny_end; typeofY = Float64)
    X_set0 = []
    Y_set0 = typeofY[]
    
    for h in altitudes
        append!(X_set0, [[i, j, Float64(h)] for j in Float64(ny_start-1)*grid_dist : grid_dist_mid : Float64(ny_end-1)*grid_dist for i in Float64(nx_start-1)*grid_dist : grid_dist_mid : Float64(nx_end-1)*grid_dist])
        append!(Y_set0, vec(Map[h][nx_start : nx_end, ny_start : ny_end]))
    end
    
    X_set = transform4GPjl(X_set0)
    Y_set = Y_set0  
    return X_set, Y_set
end

function get_Y_from_farm_location(loc, Map, grid_dist)
    idx = loc[1:2]./ grid_dist .+ 1.0
    h = loc[3]
    return Map[h][Int.(idx)...]
end
