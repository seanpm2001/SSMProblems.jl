using SSMProblems
using GaussianDistributions
using PDMats, LinearAlgebra
using Distributions
using Random
using UnPack
using StatsFuns

import AbstractMCMC: sample, AbstractSampler

## UTILITIES ###############################################################################

# GaussianDistributions.correct will error when type casting otherwise
function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
    return PDMat(Symmetric(mat))
end

# necessary for type stability of logpdf for Gaussian
function Distributions.logpdf(P::Gaussian, x)
    dP = length(P.μ)
    logdetcov = GaussianDistributions._logdet(P.Σ, dP)
    return -(
        GaussianDistributions.sqmahal(P, x) + logdetcov + dP * convert(eltype(x), log2π)
    ) / 2
end

function multinomial_resampling(
    rng::AbstractRNG, weights::AbstractVector{<:Real}, N::Integer=length(weights)
)
    return rand(rng, Distributions.Categorical(weights), N)
end

abstract type AbstractParticleContainer{T} end

# TODO: improve particle storage
struct ParticleContainer{T,WT<:Real} <: AbstractParticleContainer{T}
    vals::Vector{T}
    log_weights::Vector{WT}
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.keys(pc::ParticleContainer) = LinearIndices(pc.vals)

Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Vector{Int}) = pc.vals[i]
function Base.setindex!(pc::ParticleContainer{T}, p::T, i::Int) where {T}
    return Base.setindex!(pc.vals, p, i)
end

## LINEAR GAUSSIAN STATE SPACE MODEL #######################################################

struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics{Vector{T}}
    """
        Latent dynamics for a linear Gaussian state space model.
        The model is defined by the following equations:
        x[t] = Ax[t-1] + b + ε[t],      ε[t] ∼ N(0, Q)
    """
    A::Matrix{T}
    b::Vector{T}
    Q::PDMat{T,Matrix{T}}

    init::Gaussian{Vector{T},PDMat{T,Matrix{T}}}
end

# Convert covariance matrices to PDMats to avoid recomputing Cholesky factorizations
function LinearGaussianLatentDynamics(A::Matrix, b::Vector, Q::Matrix, init::Gaussian)
    return LinearGaussianLatentDynamics(A, b, PSDMat(Q), init)
end

function LinearGaussianLatentDynamics(
    A::Matrix{T}, Q::Matrix{T}, init::Gaussian
) where {T<:Real}
    return LinearGaussianLatentDynamics(A, zeros(T, size(A, 1)), PSDMat(Q), init)
end

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess{Vector{T}}
    """
        Observation process for a linear Gaussian state space model.
        The model is defined by the following equation:
        y[t] = Hx[t] + η[t],        η[t] ∼ N(0, R)
    """
    H::Matrix{T}
    R::PDMat{T,Matrix{T}}
end

function LinearGaussianObservationProcess(H::Matrix, R::Matrix)
    return LinearGaussianObservationProcess(H, PSDMat(R))
end

function SSMProblems.distribution(
    proc::LinearGaussianLatentDynamics{T}, extra
) where {T<:Real}
    dx = size(proc.A, 1)
    return MvNormal(proc.init.μ, proc.init.Σ)
end

function SSMProblems.distribution(
    proc::LinearGaussianLatentDynamics{T}, step::Int, state::AbstractVector{T}, extra
) where {T<:Real}
    return MvNormal(proc.A * state + proc.b, proc.Q)
end

function SSMProblems.distribution(
    proc::LinearGaussianObservationProcess{T}, step::Int, state::AbstractVector{T}, extra
) where {T<:Real}
    return MvNormal(proc.H * state, proc.R)
end

const LinearGaussianModel{T} = StateSpaceModel{
    T,D,O
} where {T,D<:LinearGaussianLatentDynamics{T},O<:LinearGaussianObservationProcess{T}}

function PSDMat(mat::AbstractMatrix)
    # this deals with rank definicient positive semi-definite matrices
    chol_mat = cholesky(mat, Val(true); check=false)
    Up = UpperTriangular(chol_mat.U[:, invperm(chol_mat.p)])
    return PDMat(mat, Cholesky(Up))
end

## FILTERING ###############################################################################

abstract type AbstractFilter <: AbstractSampler end

"""
    predict([rng,] states, model, alg, [step, extra])

propagate the filtered states forward in time.
"""
function predict end

"""
    update(states, model, data, alg, [step, extra])

update beliefs on the propagated states.
"""
function update end

"""
    initialise([rng,] model, alg, [extra])

propose an initial state distribution.
"""
function initialise end

function sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    observations::AbstractVector,
    filter::AbstractFilter;
    callback=nothing,
    kwargs...,
)
    filtered_states = initialise(rng, model, filter, nothing)
    log_evidence = zero(eltype(model))

    for t in eachindex(observations)
        proposed_states = predict(
            rng, filtered_states, model, filter, t, nothing; kwargs...
        )

        filtered_states, log_marginal = update(
            proposed_states, model, observations[t], filter, t, nothing
        )

        log_evidence += log_marginal

        callback === nothing ||
            callback(rng, model, observations, filtered_states, t; kwargs...)
    end

    return filtered_states, log_evidence
end

## KALMAN FILTER ###########################################################################

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function initialise(
    rng::AbstractRNG, model::LinearGaussianModel, filter::KalmanFilter, extras
)
    init_dist = SSMProblems.distribution(model.dyn, extras)
    return Gaussian(init_dist.μ, Matrix(init_dist.Σ))
end

function predict(
    rng::AbstractRNG,
    states::Gaussian,
    model::LinearGaussianModel,
    filter::KalmanFilter,
    step::Integer,
    extra,
)
    @unpack A, b, Q = model.dyn

    predicted_states = let μ = states.μ, Σ = states.Σ
        Gaussian(A * μ + b, A * Σ * A' + Q)
    end

    return predicted_states
end

function update(
    proposed_states::Gaussian,
    model::LinearGaussianModel,
    observation,
    filter::KalmanFilter,
    step::Integer,
    extra,
)
    @unpack H, R = model.obs

    states, residual, S = GaussianDistributions.correct(
        proposed_states, Gaussian(observation, R), H
    )

    log_marginal = logpdf(Gaussian(zero(residual), Symmetric(S)), residual)

    return states, log_marginal
end

## BOOTSTRAP FILTER ########################################################################

# TODO: fix the adaptive resampling
struct BootstrapFilter{T<:Real} <: AbstractFilter
    N::Int
    threshold::T
end

BF(N::Integer, rs::Real=1.0) = BootstrapFilter(N, rs)

resample_threshold(filter::BootstrapFilter) = filter.threshold * filter.N

function initialise(
    rng::AbstractRNG, model::StateSpaceModel, filter::BootstrapFilter, extra
)
    init_dist = SSMProblems.distribution(model.dyn, extra)
    initial_states = map(x -> rand(rng, init_dist), 1:(filter.N))

    return ParticleContainer(initial_states, zeros(eltype(model), filter.N))
end

function predict(
    rng::AbstractRNG,
    states::ParticleContainer,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    extra;
    debug::Bool=false,
)
    weights = softmax(states.log_weights)
    ess = inv(sum(abs2, weights))

    debug && println("t: $step \tESS: $ess")

    if resample_threshold(filter) ≥ ess
        idx = multinomial_resampling(rng, weights)
        fill!(states.log_weights, zero(ess))
    else
        idx = collect(1:(filter.N))
    end

    proposed_states = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x, extra), states[idx]
    )

    return ParticleContainer(proposed_states, states.log_weights)
end

function update(
    states::ParticleContainer,
    model::StateSpaceModel{T},
    observation,
    filter::BootstrapFilter,
    step::Integer,
    extra,
) where {T}
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation, extra), collect(states)
    )

    # TODO: improve this calculation
    updated_states = ParticleContainer(states.vals, states.log_weights + log_marginals)
    return (
        updated_states,
        logsumexp(updated_states.log_weights) - logsumexp(states.log_weights),
    )
end