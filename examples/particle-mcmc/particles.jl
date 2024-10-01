using StatsBase: weights, mean
using CairoMakie

include("simple-filters.jl")

## PARTICLES ###############################################################################

#=
    I want to include Frederic's original, very elegant, particle interface for reference.
    Ideally, I want to borrow the syntax and convenience of this interface such that on the
    user's end there is little to no difference between this and Lawrence Murray's particle
    path storage.

    I have yet to benchmark the differences, but I'm keen to check the performance gain on
    some DGPs.
=#

# abstract type Node{T} end

# struct Root{T} <: Node{T} end
# Root(T) = Root{T}()
# Root() = Root(Any)

# """
#     Particle{T}
# Particle as immutable LinkedList. 
# """
# struct Particle{T} <: Node{T}
#     parent::Node{T}
#     state::T
# end

# Particle(state::T) where {T} = Particle(Root(T), state)

# Base.show(io::IO, p::Particle{T}) where {T} = print(io, "Particle{$T}($(p.state))")

# """
#     linearize(particle)
# Return the trace of a particle, i.e. the sequence of states from the root to the particle.
# """
# function linearize(particle::Particle{T}) where {T}
#     trace = T[]
#     current = particle
#     while !isa(current, Root)
#         push!(trace, current.state)
#         current = current.parent
#     end
#     return trace
# end

# function linearize(f::Function, particle::Particle)
#     trace = []
#     current = particle
#     while !isa(current, Root)
#         push!(trace, f(current.state))
#         current = current.parent
#     end
#     return trace
# end

## RESAMPLERS ##############################################################################

function systematic_resampling(
    rng::AbstractRNG, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand(rng))

    # initialize sampling algorithm
    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        while v < u
            idx += 1
            v += n * weights[idx]
        end
        a[i] = idx
        u += one(u)
    end

    return a
end

# TODO: this should be done in the log domain and also parallelized
function metropolis_resampling(
    rng::AbstractRNG, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    bins = Int64(cld(mean(weights), maximum(weights)))

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        k = i
        for _ in 1:bins
            j = rand(rng, 1:n)
            v = weights[j] / weights[k]
            if rand(rng) ≤ v
                k = j
            end
        end
        a[i] = k
    end

    return a
end

## JACOB-MURRAY PARTICLE STORAGE ###########################################################

mutable struct ParticleTree{T}
    states::Vector{T}
    parents::Vector{Int64}
    leaves::Vector{Int64}
    offspring::Vector{Int64}

    function ParticleTree(states::Vector{T}, M::Integer) where {T}
        nodes = Vector{T}(undef, M)
        @inbounds nodes[1:length(states)] = states
        return new{T}(nodes, zeros(Int64, M), 1:length(states), zeros(Int64, M))
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
            j = tree.parents[j]
            if j > 0
                tree.offspring[j] -= 1
            end
        end
    end
end

function insert!(
    tree::ParticleTree{T}, states::Vector{T}, a::AbstractVector{Int64}; debug::Bool=false
) where {T}
    ## parents of new generation
    parents = getindex(tree.leaves, a)

    ## ensure there are enough dead branches
    if (sum(iszero.(tree.offspring)) < length(a))
        debug && print("\texpanding tree")
        expand!(tree)
    end

    ## find dead branches and expand if too few branches
    z = cumsum(iszero.(tree.offspring))
    @inbounds for i in eachindex(states)
        # TODO: make a better lower bound operation
        tree.leaves[i] = minimum([j for j in eachindex(tree) if (z[j] ≥ i)])
    end

    ## insert new generation and update parent child relationships
    setindex!(tree.states, states, tree.leaves)
    return setindex!(tree.parents, parents, tree.leaves)
end

# TODO: clean this up
function expand!(tree::ParticleTree)
    M = length(tree)
    resize!(tree.states, 2 * M)

    # new allocations must be zero valued, this is not a perfect solution
    tree.parents = [tree.parents; zero(tree.parents)]
    return tree.offspring = [tree.offspring; zero(tree.offspring)]
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

# replaces the function defined in simple-filters.jl
function initialise(
    rng::AbstractRNG, model::StateSpaceModel, filter::BootstrapFilter, extra
)
    init_dist = SSMProblems.distribution(model.dyn, extra)
    initial_states = map(x -> rand(rng, init_dist), 1:(filter.N))

    return AncestryContainer(initial_states, zeros(eltype(model), filter.N))
end

function predict(
    rng::AbstractRNG,
    states::AncestryContainer,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    extra;
    debug::Bool=false,
)
    weights = softmax(states.log_weights)
    ess = inv(sum(abs2, weights))

    debug && print("\nt: $step \tESS: $ess")

    if resample_threshold(filter) ≥ ess
        idx = multinomial_resampling(rng, weights)
        fill!(states.log_weights, zero(ess))
    else
        idx = 1:(filter.N)
    end

    proposed_states = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x, extra), getindex(states, idx)
    )

    prune!(states.tree, get_offspring(idx))
    insert!(states.tree, proposed_states, idx; debug)

    return states
end

function update(
    states::AncestryContainer,
    model::StateSpaceModel{T},
    observation,
    filter::BootstrapFilter,
    step::Integer,
    extra,
) where {T}
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation, extra), collect(states)
    )

    prev_log_marginal = logsumexp(states.log_weights)
    states.log_weights += log_marginals
    return (states, logsumexp(states.log_weights) - prev_log_marginal)
end

## EXTRACTING ACESTRY ######################################################################

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

## VISUALIZATION ###########################################################################

# use a local level trend model
function simulation_model(σx²::T, σy²::T) where {T<:Real}
    init = Gaussian(zeros(T, 2), PDMat(diagm(ones(T, 2))))
    dyn = LinearGaussianLatentDynamics(T[1 1; 0 1], T[0; 0], [σx² 0; 0 0], init)
    obs = LinearGaussianObservationProcess(T[1 0], [σy²;;])
    return StateSpaceModel(dyn, obs)
end

# generate a model
true_params = randexp(Float32, 2);
true_model = simulation_model(true_params...);

# simulate data
rng = MersenneTwister(1234);
_, _, data = sample(rng, true_model, 150);

# test the adaptive resampling procedure
hist, llbf = sample(rng, true_model, data, BF(512, 0.5); debug=true);

# plot the smoothed states to validate the algorithm
smoothed_trend = begin
    fig = Figure()
    ax = Axis(fig[1, 1])

    # this is gross but it works fro visualization purposes
    all_paths = map(x -> hcat(x...), get_ancestry(hist.tree))
    mean_paths = mean(all_paths, weights(softmax(hist.log_weights)))

    # plot smoothed states in black and observed data in red
    lines!(ax, mean_paths[1, :]; color=:black)
    lines!(ax, vcat(data...); color=:red)

    fig
end