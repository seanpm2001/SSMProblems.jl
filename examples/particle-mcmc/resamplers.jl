using Random
using Distributions

function multinomial_resampling(
    rng::AbstractRNG, weights::AbstractVector{WT}, n::Int64=length(weights); kwargs...
) where {WT<:Real}
    return rand(rng, Distributions.Categorical(weights), n)
end

function systematic_resampling(
    rng::AbstractRNG, weights::AbstractVector{WT}, n::Int64=length(weights); kwargs...
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
    rng::AbstractRNG,
    weights::AbstractVector{WT},
    n::Int64=length(weights);
    ε::Float64=0.01,
    kwargs...,
) where {WT<:Real}
    # pre-calculations
    β = mean(weights)
    bins = Int64(cld(log(ε), log(1 - β)))

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

# TODO: this should be done in the log domain and also parallelized
function rejection_resampling(
    rng::AbstractRNG, weights::AbstractVector{WT}, n::Int64=length(weights); kwargs...
) where {WT<:Real}
    # pre-calculations
    max_weight = maximum(weights)

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        j = i
        u = rand(rng)
        while u > weights[j] / max_weight
            j = rand(1:n)
            u = rand(rng)
        end
        a[i] = j
    end

    return a
end
