using SSMProblems
using GaussianDistributions
using PDMats, LinearAlgebra
using Distributions
using Random
using UnPack
using StatsFuns

import AbstractMCMC: sample, AbstractSampler

## UTILITIES ##################################################################

# GaussianDistributions.correct will error when type casting otherwise
function Base.convert(::Type{PDMat{T, MT}}, mat::MT) where {MT<:AbstractMatrix, T<:Real}
    return PDMat(Symmetric(mat))
end

# necessary for type stability of logpdf for Gaussian
function Distributions.logpdf(P::Gaussian, x)
    dP = length(P.μ)
    logdetcov = GaussianDistributions._logdet(P.Σ, dP)
    return -(GaussianDistributions.sqmahal(P,x) + logdetcov + dP*convert(eltype(x), log2π))/2
end

function multinomial_resampling(
        rng::AbstractRNG,
        weights::AbstractVector{<:Real},
        N::Integer = length(weights)
    )
    return rand(rng, Distributions.Categorical(weights), N)
end

# TODO: improve particle storage
struct ParticleContainer{T, WT<:Real}
    vals::Vector{T}
    log_weights::Vector{WT}
end

Base.collect(pc::ParticleContainer) = pc.vals
Base.length(pc::ParticleContainer) = length(pc.vals)
Base.keys(pc::ParticleContainer) = LinearIndices(pc.vals)

Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Int) = pc.vals[i]
Base.@propagate_inbounds Base.getindex(pc::ParticleContainer, i::Vector{Int}) = pc.vals[i]
Base.setindex!(pc::ParticleContainer{T}, p::T, i::Int) where T = Base.setindex!(pc.vals, p, i)

## LINEAR GAUSSIAN STATE SPACE MODEL ##########################################

struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics
    """
        Latent dynamics for a linear Gaussian state space model.
        The model is defined by the following equations:
        x[t] = Ax[t-1] + ε[t],      ε[t] ∼ N(0, Q)
    """
    A::Matrix{T}
    Q::PDMat{T, Matrix{T}}
end

# Convert covariance matrices to PDMats to avoid recomputing Cholesky factorizations
function LinearGaussianLatentDynamics(A::Matrix, Q::Matrix)
    return LinearGaussianLatentDynamics(A, PDMat(Q))
end

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess
    """
        Observation process for a linear Gaussian state space model.
        The model is defined by the following equation:
        y[t] = Hx[t] + η[t],        η[t] ∼ N(0, R)
    """
    H::Matrix{T}
    R::PDMat{T, Matrix{T}}
end

function LinearGaussianObservationProcess(B::Matrix, R::Matrix)
    return LinearGaussianObservationProcess(B, PDMat(R))
end

function SSMProblems.distribution(
        proc::LinearGaussianLatentDynamics{T},
        extra
    ) where {T<:Real}
        dx = size(proc.A, 1)
    return MvNormal(zeros(T, dx), diagm(ones(T, dx)))
end

function SSMProblems.distribution(
        proc::LinearGaussianLatentDynamics{T},
        step::Int,
        state::AbstractVector{T},
        extra
    ) where {T<:Real}
    return MvNormal(proc.A*state, proc.Q)
end

function SSMProblems.distribution(
        proc::LinearGaussianObservationProcess{T},
        step::Int,
        state::AbstractVector{T},
        extra
    ) where {T<:Real}
    return MvNormal(proc.H*state, proc.R)
end

const LinearGaussianModel{T<:Real} = StateSpaceModel{
    LinearGaussianLatentDynamics{T},
    LinearGaussianObservationProcess{T}
}

Base.eltype(::LinearGaussianModel{T}) where T = T
Base.eltype(::StateSpaceModel) = error("model element type must be explicit")

## FILTERING ##################################################################

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
    prior([rng,] model, alg, [extra])

propose an initial state distribution.
"""
function prior end

function sample(
        rng::AbstractRNG,
        model::StateSpaceModel,
        observations::AbstractVector,
        filter::AbstractFilter
    )
    MT = eltype(model)
    
    filtered_states = prior(rng, model, filter, nothing)
    log_evidence = zero(MT)

    for t in eachindex(observations)
        proposed_states = predict(
            rng, filtered_states, model, filter, t, nothing
        )

        filtered_states, log_marginal = update(
            proposed_states, model, observations[t], filter, t, nothing
        )

        log_evidence += log_marginal
    end

    return filtered_states, log_evidence
end

## KALMAN FILTER ##############################################################

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function prior(
        rng::AbstractRNG,
        model::LinearGaussianModel,
        filter::KalmanFilter,
        extras
    )
    init_dist = SSMProblems.distribution(model.dyn, extras)
    return Gaussian(init_dist.μ, Matrix(init_dist.Σ))
end

function predict(
        rng::AbstractRNG,
        particles::Gaussian,
        model::LinearGaussianModel,
        filter::KalmanFilter,
        step::Integer,
        extra
    )
    @unpack A, Q = model.dyn

    predicted_particles = let μ = particles.μ, Σ = particles.Σ
        Gaussian(A*μ, A*Σ*A' + Q)
    end

    return predicted_particles
end

function update(
        proposed_particles::Gaussian,
        model::LinearGaussianModel,
        observation,
        filter::KalmanFilter,
        step::Integer,
        extra
    )
    @unpack H, R = model.obs

    particles, residual, S = GaussianDistributions.correct(
        proposed_particles,
        Gaussian(observation, R), H
    )

    log_marginal = logpdf(
        Gaussian(zero(residual), Symmetric(S)),
        residual
    )

    return particles, log_marginal
end

## BOOTSTRAP FILTER ###########################################################

# TODO: adaptive resampling, set by default to always resmaple
struct BootstrapFilter <: AbstractFilter
    N::Int64
    threshold::Float64
end

BF(N::Integer) = BootstrapFilter(N, 1.0)

resample_threshold(filter::BootstrapFilter) = filter.threshold*filter.N

function prior(
        rng::AbstractRNG,
        model::StateSpaceModel,
        filter::BootstrapFilter,
        extra
    )
    init_dist = SSMProblems.distribution(model.dyn, extra)
    initial_states = map(
        x -> rand(rng, init_dist),
        1:filter.N
    )

    return ParticleContainer(initial_states, zeros(eltype(model), filter.N))
end

function predict(
        rng::AbstractRNG,
        particles::ParticleContainer,
        model::StateSpaceModel,
        filter::BootstrapFilter,
        step::Integer,
        extra
    )
    weights = softmax(particles.log_weights)
    idx = multinomial_resampling(rng, weights)

    proposed_states = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x, extra),
        particles[idx]
    )

    return ParticleContainer(proposed_states, particles.log_weights)
end

function update(
        particles::ParticleContainer,
        model::StateSpaceModel,
        observation,
        filter::BootstrapFilter,
        step::Integer,
        extra
    )
    log_marginals = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation, extra),
        collect(particles)
    )

    return (
        ParticleContainer(particles.vals, log_marginals),
        logsumexp(log_marginals) - convert(eltype(model), log(filter.N))
    )
end