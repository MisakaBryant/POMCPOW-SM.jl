# function obsSimilar(::POMDP{S,A,O}, ::O, ::O, similarity_threshold::Float64) where {S,A,O}
#     return o1 == o2
# end

# function findSimilarKey(pomdp::POMDP, dict::Dict, k, similarity_threshold::Float64)
#     for key in keys(dict)
#         if obsSimilar(pomdp, key[2], k[2], similarity_threshold)
#             return key
#         end
#     end
#     return nothing
# end

function beliefSimilar(b1::POWNodeBelief, b2::POWNodeBelief; similarity_threshold::Float64 = 0.9)
    # 1. 提取粒子集合
    particles1 = b1.dist.items  # 粒子集合 1
    particles2 = b2.dist.items  # 粒子集合 2
    # 2. 统计每个状态的频率，并构造频率向量
    freq1 = Dict()
    freq2 = Dict()
    for p in particles1
        freq1[p] = get(freq1, p, 0) + 1
    end
    for p in particles2
        freq2[p] = get(freq2, p, 0) + 1
    end
    # 3. 获取所有可能的状态集合
    all_states = union(keys(freq1), keys(freq2))
    # 4. 构造频率向量
    v1 = [get(freq1, s, 0) for s in all_states]
    v2 = [get(freq2, s, 0) for s in all_states]
    # 5. 计算余弦相似度
    dot_product = dot(v1, v2)  # 向量点积
    norm1 = norm(v1)           # 向量 1 的范数
    norm2 = norm(v2)           # 向量 2 的范数
    # 避免除以零
    if norm1 == 0.0 || norm2 == 0.0
        return 0.0  # 如果任一向量为空，返回相似度为 0
    end
    similarity = dot_product / (norm1 * norm2)
    # 6. 判断是否超过相似度阈值
    return similarity >= similarity_threshold
end

function simulate(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}
    

    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0
    end

    # 动作选择 ACTIONPROGWIDEN
    if sol.enable_action_pw
        total_n = tree.total_n[h]
        if length(tree.tried[h]) <= sol.k_action*total_n^sol.alpha_action
            if h == 1
                a = next_action(pomcp.next_action, pomcp.problem, tree.root_belief, POWTreeObsNode(tree, h))
            else
                a = next_action(pomcp.next_action, pomcp.problem, StateBelief(tree.sr_beliefs[h]), POWTreeObsNode(tree, h))
            end
            if !sol.check_repeat_act || !haskey(tree.o_child_lookup, (h,a))
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            sol.check_repeat_act)
            end
        end
    else # run through all the actions
        if isempty(tree.tried[h])
            if h == 1
                action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
            else
                action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
            end
            anode = length(tree.n)
            for a in action_space_iter
                push_anode!(tree, h, a,
                            init_N(pomcp.init_N, pomcp.problem, POWTreeObsNode(tree, h), a),
                            init_V(pomcp.init_V, pomcp.problem, POWTreeObsNode(tree, h), a),
                            false)
            end
        end
    end
    total_n = tree.total_n[h]

    best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
    a = tree.a_labels[best_node]
    # end ACTIONPROGWIDEN



    # 扩展树
    new_node = false

    if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)
        # 观测节点数量未达上限（上限渐进拓宽）
        # G
        sp, o, r = @gen(:sp, :o, :r)(pomcp.problem, s, a, sol.rng)
        # key = findSimilarKey(pomcp.problem, tree.a_child_lookup, (best_node,o), sol.similarity_threshold)
        if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node,o))
            # 重复的观测节点索引
            hao = tree.a_child_lookup[(best_node, o)]
        else
            # 新的观测，新的观测节点
            new_node = true
            # hao为新的观测节点索引
            hao = length(tree.sr_beliefs) + 1
            
            # TODO: 更新信念粒子集，粒子数量50
            if h != 1
                b = tree.sr_beliefs[h]
            else
                b = tree.root_belief
            end
            updated_belief = belief_update(pomcp.node_sr_belief_updater, 
                                            pomcp.problem, b, a, o, 10, sol.rng)

            # TODO: 判断信念状态是否相似
            similar_belief = false
            if sol.check_repeat_obs
                for pair in tree.generated[best_node]
                    if beliefSimilar(updated_belief, tree.sr_beliefs[pair.second]; similarity_threshold=sol.similarity_threshold)
                        hao = pair.second
                        new_node = false
                        similar_belief = true
                        break
                    end
                end
            end

            if !similar_belief
                push!(tree.sr_beliefs, updated_belief)

                push!(tree.total_n, 0)
                push!(tree.tried, Int[])
                push!(tree.o_labels, o)

                if sol.check_repeat_obs
                    tree.a_child_lookup[(best_node, o)] = hao
                end
                tree.n_a_children[best_node] += 1
            end
        end
        push!(tree.generated[best_node], o=>hao)
    else
        # @info "Action node $best_node has too many children, not expanding."
        # G，不需要新的观测
        sp, r = @gen(:sp, :r)(pomcp.problem, s, a, sol.rng)

    end

    if r == Inf
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end

    if new_node
        # ROLLOUT
        R = r + POMDPs.discount(pomcp.problem)*estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, hao), d-1)
    else
        pair = rand(sol.rng, tree.generated[best_node])
        o = pair.first
        hao = pair.second

        # 采样一个状态
        sp, r = rand(sol.rng, tree.sr_beliefs[hao])
        R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
    end

    tree.n[best_node] += 1
    tree.total_n[h] += 1
    if tree.v[best_node] != -Inf
        tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
    end

    return R
end



