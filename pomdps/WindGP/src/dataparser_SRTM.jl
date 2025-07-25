# Parses the data provided in ../data/SRTM/

function get_elevation_data(farm, srtm_coord, grid_dist_mid, grid_dist, nx_start, nx_end, ny_start, ny_end)

    remove_altitude(x) = x[1:2]
    filename_t(h) = "custom_wind-speed_$(h)m.xyz"
    datapath_t(loc) = "../../../WindGP/data/GWA/$(loc)/"
    srtmpath = "../../../WindGP/data/SRTM/"
    srtmname_t(c) = "$(c).xyz"
    datapath = datapath_t(farm)
    srtmname = srtmname_t(srtm_coord)
    
    h = 100    # can be any valid altitude
    data_3D = Dict{Int, Array{Float64,2}}()
    
    filename = filename_t(h)
    D = data_2D(datapath, filename)
    Dxy = reshape(collect(zip(D.x, D.y)), length(unique(D.x)), length(unique(D.y)))

    Farm_coords, Earth_coords = get_dataset(Dict(h => Dxy), [h], grid_dist_mid, grid_dist, nx_start, nx_end, ny_start, ny_end; typeofY = Tuple{Float64,Float64})
    Earth_coords = transform4GPjl(tuple_to_array.(Earth_coords); dim=2)
    Farm_coords = transform4GPjl(remove_altitude.(eachcol(Farm_coords)); dim=2)
        
    SRTM_data = data_2D(srtmpath, srtmname)
    SRTM_coords = transform4GPjl(tuple_to_array.(collect(zip(SRTM_data.x, SRTM_data.y))); dim=2)
    kdtree_SRTM_coords = NearestNeighbors.KDTree(SRTM_coords)

    nn = knn.(Ref(kdtree_SRTM_coords), eachcol(Earth_coords), Ref(1))
    nn = [item[1][1] for item in nn]
    elevation = SRTM_data.value[nn]

    return elevation
end
