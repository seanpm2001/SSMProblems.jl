using Random
using Distributions

abstract type AbstractResampler end

## DOUBLE PRECISION STABLE ALGORITHMS ######################################################

struct Multinomial <: AbstractResampler end

function resample(
    rng::AbstractRNG, ::Multinomial, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    return rand(rng, Distributions.Categorical(weights), n)
end

struct Systematic <: AbstractResampler end

function resample(
    rng::AbstractRNG, ::Systematic, weights::AbstractVector{WT}, n::Int64=length(weights)
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

function resample(
    rng::AbstractRNG,
    alg::Systematic,
    weights::AbstractVector{Float32},
    n::Int64=length(weights),
)
    try
        return resample(rng, alg, weights, n)
    catch e
        throw(e("Systematic resampling is not numerically stable for single precision"))
    end
end

## SINGLE PRECISION STABLE ALGORITHMS ######################################################

struct Metropolis{T<:Real} <: AbstractResampler
    ε::T
    function Metropolis(ε::T=0.01) where {T<:Real}
        return new{T}(ε)
    end
end

# TODO: this should be done in the log domain and also parallelized
function resample(
    rng::AbstractRNG,
    resampler::Metropolis,
    weights::AbstractVector{WT},
    n::Int64=length(weights);
) where {WT<:Real}
    # pre-calculations
    β = mean(weights)
    B = Int64(cld(log(resampler.ε), log(1 - β)))

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        k = i
        for _ in 1:B
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

struct Rejection <: AbstractResampler end

# TODO: this should be done in the log domain and also parallelized
function resample(
    rng::AbstractRNG, ::Rejection, weights::AbstractVector{WT}, n::Int64=length(weights)
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
