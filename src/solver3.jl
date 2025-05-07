function obsSimilar(::POMDP{S,A,O}, ::O, ::O, similarity_threshold::Float64) where {S,A,O}
    return o1 == o2
end

function findSimilarKey(pomdp::POMDP, dict::Dict, k, similarity_threshold::Float64)
    for key in keys(dict)
        if obsSimilar(pomdp, key[2], k[2], similarity_threshold)
            return key
        end
    end
    return nothing
end

function simplifyObs(::POMDP{S,A,O}, o::O) where {S,A,O}
    return o
end

function simulate(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, d) where {B,S,A,O}

    tree = h_node.tree
    h = h_node.node

    sol = pomcp.solver

    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0
    end

    tree.total_n[h] += 1

    # 如果是叶节点
    if isempty(tree.tried[h])
        total_n = tree.total_n[h]
        if h == 1
            action_space_iter = POMDPs.actions(pomcp.problem, tree.root_belief)
        else
            action_space_iter = POMDPs.actions(pomcp.problem, StateBelief(tree.sr_beliefs[h]))
        end
        if sol.enable_action_pw
            width = min(sol.k_action*total_n^sol.alpha_action, length(action_space_iter))
        else
            width = length(action_space_iter)
        end
        # 宽度优先，插入所有动作
        while length(tree.tried[h]) < width
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
        rewards = []
        # 探索所有动作
        for node in tree.tried[h]
            push!(rewards, simulateActionLeaf(pomcp, h_node, s, node, d))
        end
        R = mean(rewards)
    else
        # 选择最佳动作
        best_node = select_best(pomcp.criterion, h_node, pomcp.solver.rng)
        a = tree.a_labels[best_node]

        # 扩展树
        new_node = false
        if tree.n_a_children[best_node] <= sol.k_observation*(tree.n[best_node]^sol.alpha_observation)
            # 观测节点数量未达上限（上限渐进拓宽）
            # G
            sp, o, r = @gen(:sp, :o, :r)(pomcp.problem, s, a, sol.rng)
            o = simplifyObs(pomcp.problem, o)
            # key = findSimilarKey(pomcp.problem, tree.a_child_lookup, (best_node,o), sol.similarity_threshold)
            if sol.check_repeat_obs && haskey(tree.a_child_lookup, (best_node, o))
                # 重复的观测节点索引
                hao = tree.a_child_lookup[(best_node, o)]
            else
                # 新的观测，新的观测节点
                new_node = true
                # hao为新的观测节点索引
                hao = length(tree.sr_beliefs) + 1
                push!(tree.sr_beliefs,
                    init_node_sr_belief(pomcp.node_sr_belief_updater,
                                        pomcp.problem, s, a, sp, o, r))
                push!(tree.total_n, 0)
                push!(tree.tried, Int[])
                push!(tree.o_labels, o)

                if sol.check_repeat_obs
                    tree.a_child_lookup[(best_node, o)] = hao
                end
                tree.n_a_children[best_node] += 1
            end
            push!(tree.generated[best_node], o=>hao)
        else
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
            # 更新信念点，append sp to B(hao);  append Z(o | s, a, sp ) to W(hao)
            push_weighted!(tree.sr_beliefs[hao], pomcp.node_sr_belief_updater, s, sp, r)
            # 采样一个状态
            sp, r = rand(sol.rng, tree.sr_beliefs[hao])
            R = r + POMDPs.discount(pomcp.problem)*simulate(pomcp, POWTreeObsNode(tree, hao), sp, d-1)
        end

        tree.n[best_node] += 1
        if tree.v[best_node] != -Inf
            tree.v[best_node] += (R-tree.v[best_node])/tree.n[best_node]
        end
    end
    

    

    return R
end

# 仿真动作叶节点
function simulateActionLeaf(pomcp::POMCPOWPlanner, h_node::POWTreeObsNode{B,A,O}, s::S, node::Int64, d) where {B,S,A,O}
    tree = h_node.tree
    a = tree.a_labels[node]
    h = h_node.node
    sol = pomcp.solver
    if POMDPs.isterminal(pomcp.problem, s) || d <= 0
        return 0.0
    end
    sp, r = @gen(:sp, :r)(pomcp.problem, s, a, sol.rng)
    if r == Inf
        @warn("POMCPOW: +Inf reward. This is not recommended and may cause future errors.")
    end
    R = r + POMDPs.discount(pomcp.problem)*estimate_value(pomcp.solved_estimate, pomcp.problem, sp, POWTreeObsNode(tree, h), d-1)
    tree.n[node] += 1
    tree.v[node] += (R-tree.v[node])/tree.n[node]
    return R
    
end