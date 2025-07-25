function show_results_as_dataframe(csv_filenames; normalizer_rewards = 1.0e6)

    if isempty(csv_filenames) return end

    μs = []
    σs = []
    pr = []
    tt = []
    td = []
    solvername = ""

    for fl in csv_filenames
        file_path = csv_dir * fl
        parsed_data = readdlm(file_path)
        no_of_runs = size(parsed_data, 1)    # no of lines in file
        
        solvername = parsed_data[1,1]
        layouttype = parsed_data[1,2]
        params = parsed_data[1,2:end-2]
        
        times_taken = parsed_data[:, end-1]
        rewards = parsed_data[:, end] / normalizer_rewards

        avg_rewards = mean(rewards)
        std_rewards = std(rewards)./ sqrt(no_of_runs)

        avg_times = mean(times_taken)
        std_times = std(times_taken)./ sqrt(no_of_runs)

        push!(μs, avg_rewards)
        push!(σs, std_rewards)
        push!(pr, params)
        push!(tt, avg_times)
        push!(td, std_times)
    end

    # Sort all 
    srt = sortperm(pr)
    μs = μs[srt]
    σs = σs[srt]
    pr = pr[srt]
    tt = tt[srt]
    td = td[srt]

    Data_results = DataFrame(Params = pr,
    Reward = μs,
    Reward_pm = σs, 
    Time = tt,
    Time_pm = td                          
    );
    
    # @show solvername
    @show Data_results
    println("\n")
end

function get_csv_filenames(files_in_dir, val)
    idx = findall(x -> occursin(val, x), files_in_dir)
    return files_in_dir[idx]
end

function parse_results(layouttypes, csv_dir)
    rd = readdir(csv_dir)

    for sv in layouttypes
        csv_filenames_sv  = get_csv_filenames(rd, String(sv))
        show_results_as_dataframe(csv_filenames_sv)
    end
end