using DataStructures: Stack

## PARTICLES ###############################################################################

abstract type AbstractParticleContainer{T} end

"""
    store!(particles, new_states, [idx])

update the state component of the particle container, with optional parent indices supplied
for use in ancestry storage.
"""
function store! end

"""
    reset_weights!(particles)

in-place method to reset the log weights of the particle cloud to zero; typically called
following a resampling step.
"""
function reset_weights! end

mutable struct ParticleContainer{T,WT<:Real} <: AbstractParticleContainer{T}
    vals::Vector{T}
    log_weights::Vector{WT}
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.keys(pc::ParticleContainer) = LinearIndices(pc.vals)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Vector{Int}) = pc.vals[i]

function store!(pc::ParticleContainer, new_states, idx...; kwargs...)
    setindex!(pc.vals, new_states, eachindex(pc))
    return pc
end

function reset_weights!(pc::ParticleContainer{T,WT}) where {T,WT<:Real}
    fill!(pc.log_weights, zero(WT))
    return pc.log_weights
end

## JACOB-MURRAY PARTICLE STORAGE ###########################################################

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
    ## insert new offspring counts
    setindex!(tree.offspring, offspring, tree.leaves)

    ## update each branch
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
    ## parents of new generation
    parents = getindex(tree.leaves, a)

    ## ensure there are enough dead branches
    if (length(tree.free_indices) < length(a))
        @debug "expanding tree"
        expand!(tree)
    end

    ## find places for new states
    @inbounds for i in eachindex(states)
        tree.leaves[i] = pop!(tree.free_indices)
    end

    ## insert new generation and update parent child relationships
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

## FILTERING WITH ANCESTRY #################################################################

mutable struct AncestryContainer{T,WT<:Real} <: AbstractParticleContainer{T}
    tree::ParticleTree{T}
    log_weights::Vector{WT}

    function AncestryContainer(
        initial_states::Vector{T}, log_weights::Vector{WT}, C::Int64=1
    ) where {T,WT<:Real}
        N = length(log_weights)
        M = floor(C * N * log(N))
        tree = ParticleTree(initial_states, Int64(M))
        return new{T,WT}(tree, log_weights)
    end
end

function Base.collect(ac::AncestryContainer)
    return getindex(ac.tree.states, ac.tree.leaves)
end

function Base.getindex(ac::AncestryContainer, a::AbstractVector{Int64})
    return getindex(ac.tree.states, getindex(ac.tree.leaves, a))
end

function reset_weights!(ac::AncestryContainer{T,WT}) where {T,WT<:Real}
    fill!(ac.log_weights, zero(WT))
    return ac.log_weights
end

function store!(ac::AncestryContainer, new_states, idx)
    prune!(ac.tree, get_offspring(idx))
    insert!(ac.tree, new_states, idx)
    return ac
end

# start at each leaf and retrace it's steps to the root node
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
