using DataStructures: Stack
using StatsBase

## PARTICLES ###############################################################################

mutable struct ParticleContainer{T,WT<:Real}
    filtered::Vector{T}
    proposed::Vector{T}
    ancestors::Vector{Int64}
    log_weights::Vector{WT}

    function ParticleContainer(
        initial_states::Vector{T}, log_weights::Vector{WT}
    ) where {T,WT<:Real}
        return new{T,WT}(
            initial_states, similar(initial_states), eachindex(log_weights), log_weights
        )
    end
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.keys(pc::ParticleContainer) = LinearIndices(pc.vals)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Vector{Int}) = pc.vals[i]

function reset_weights!(pc::ParticleContainer{T,WT}) where {T,WT<:Real}
    fill!(pc.log_weights, zero(WT))
    return pc.log_weights
end

function StatsBase.weights(pc::ParticleContainer)
    return softmax(pc.log_weights)
end

## SPARSE PARTICLE STORAGE #################################################################

Base.append!(s::Stack, a::AbstractVector) = map(x -> push!(s, x), a)

mutable struct ParticleTree{T}
    states::Vector{T}
    parents::Vector{Int64}
    leaves::Vector{Int64}
    offspring::Vector{Int64}
    free_indices::Stack{Int64}

    function ParticleTree(states::Vector{T}, M::Integer) where {T}
        nodes = Vector{T}(undef, M)
        initial_free_indices = Stack{Int64}()
        append!(initial_free_indices, M:-1:(length(states) + 1))
        @inbounds nodes[1:length(states)] = states
        return new{T}(
            nodes, zeros(Int64, M), 1:length(states), zeros(Int64, M), initial_free_indices
        )
    end
end

Base.length(tree::ParticleTree) = length(tree.states)
Base.keys(tree::ParticleTree) = LinearIndices(tree.states)

function prune!(tree::ParticleTree, offspring::Vector{Int64})
    # insert new offspring counts
    setindex!(tree.offspring, offspring, tree.leaves)

    # update each branch
    @inbounds for i in eachindex(offspring)
        j = tree.leaves[i]
        while (j > 0) && (tree.offspring[j] == 0)
            push!(tree.free_indices, j)
            j = tree.parents[j]
            if j > 0
                tree.offspring[j] -= 1
            end
        end
    end
end

function insert!(
    tree::ParticleTree{T}, states::Vector{T}, a::AbstractVector{Int64}
) where {T}
    # parents of new generation
    parents = getindex(tree.leaves, a)

    # ensure there are enough dead branches
    if (length(tree.free_indices) < length(a))
        @debug "expanding tree"
        expand!(tree)
    end

    # find places for new states
    @inbounds for i in eachindex(states)
        tree.leaves[i] = pop!(tree.free_indices)
    end

    # insert new generation and update parent child relationships
    setindex!(tree.states, states, tree.leaves)
    setindex!(tree.parents, parents, tree.leaves)
    return tree
end

function expand!(tree::ParticleTree)
    M = length(tree)
    resize!(tree.states, 2 * M)

    # new allocations must be zero valued, this is not a perfect solution
    tree.parents = [tree.parents; zero(tree.parents)]
    tree.offspring = [tree.offspring; zero(tree.offspring)]
    append!(tree.free_indices, (2 * M):-1:(M + 1))
    return tree
end

function get_offspring(a::AbstractVector{Int64})
    offspring = zero(a)
    for i in a
        offspring[i] += 1
    end
    return offspring
end

function get_ancestry(tree::ParticleTree{T}) where {T}
    paths = Vector{Vector{T}}(undef, length(tree.leaves))
    @inbounds for (k, i) in enumerate(tree.leaves)
        j = tree.parents[i]
        xi = tree.states[i]

        xs = [xi]
        while j > 0
            push!(xs, tree.states[j])
            j = tree.parents[j]
        end
        paths[k] = reverse(xs)
    end
    return paths
end
