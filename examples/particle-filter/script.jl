# # Partilce Filter with adaptive resampling
using StatsBase
using AbstractMCMC
using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns
using Metal

# Filter
ess(weights) = inv(sum(abs2, weights))
get_weights(logweights::T) where {T<:AbstractVector{<:Real}} = StatsFuns.softmax(logweights)
logZ(arr::AbstractArray) = StatsFuns.logsumexp(arr)

function resample_systematic(
    rng::Random.AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    # check input
    m = length(weights)
    m > 0 || error("weight vector is empty")

    # pre-calculations
    @inbounds v = n * weights[1]
    u = oftype(v, rand(rng))

    # find all samples
    samples = Array{Int64}(undef, n)
    sample = 1
    @inbounds for i in 1:n
        # as long as we have not found the next sample
        while v < u
            # increase and check the sample
            sample += 1
            sample > m &&
                error("sample could not be selected (are the weights normalized?)")

            # update the cumulative sum of weights (scaled by `n`)
            v += n * weights[sample]
        end

        # save the next sample
        samples[i] = sample

        # update `u`
        u += one(u)
    end

    return samples
end

"""
        resample(rng::AbstractRNG, weights::AbstractArray, particles::AbstractArray)

Resample `particles` in-place
"""
function resample!(rng::AbstractRNG, weights::AbstractArray, particles::AbstractArray)
    idx = resample_systematic(rng, weights);
    num_resamples = zeros(length(idx))
    for i in idx
        num_resamples[i] += 1
    end
    removed = findall(num_resamples .== 0)

    for (i, num_children) in enumerate(num_resamples)
        if num_children > 1
            for _ in 2:num_children
                j = popfirst!(removed)
                particles[j] = particles[i]
            end
        end
    end
end

"""
        filter(rng::AbstractRNG, model::StateSpaceModel, N::Int, observations::AbstractArray, threshold::Real)

Estimate log-evidence using `N` particles. Resample particles when ESS falls below `N * threshold`.
"""
function filter(
    rng::AbstractRNG,
    model::StateSpaceModel,
    N::Int,
    observations::AbstractArray{T},
    threshold::Real=0.5
) where T <: Real

    gpu_state = Metal.zeros(N; storage=Metal.Shared)
    gpu_logweights = Metal.zeros(N; storage=Metal.Shared)

    # Use unified memory option to avoid moving states and weights back and forth from the GPU
    cpu_state = unsafe_wrap(Array{Float32}, gpu_state, size(gpu_state))
    cpu_logweights = unsafe_wrap(Array{Float32}, gpu_logweights, size(gpu_logweights))

    logevidence = 0
    for (step, observation) in enumerate(observations)
        weights = get_weights(cpu_logweights)
        if ess(weights) <= threshold * N
            resample!(rng, weights, cpu_state)
            fill!(cpu_logweights, 0.)
        end

        logZ0 = logZ(cpu_logweights)
        Metal.@sync simulate!(model.dyn, step, gpu_state, nothing)
        Metal.@sync logdensity!(model.obs, gpu_logweights, step, gpu_state, observation, nothing)
        logZ1 = logZ(cpu_logweights)

        logevidence += logZ1 - logZ0
    end
    return logevidence
end

# Model definition
struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics
    σ::T
end

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess
    σ::T
end

const LinearGaussianSSM{T} = StateSpaceModel{
    <:LinearGaussianLatentDynamics{T},<:LinearGaussianObservationProcess{T}
};

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics{T},
    extra::Nothing
) where T <: Real
    return Normal{T}(T(0), dyn.σ)
end

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics{T},
    step::Int,
    state::Real,
    extra::Nothing
) where T <: Real
    return Normal{T}(state, dyn.σ)
end

function SSMProblems.distribution(
    obs::LinearGaussianObservationProcess{T},
    step::Int,
    state::Real,
    extra::Nothing
) where T <: Real
    return Normal{T}(state, dyn.σ)
end

function simulate!(
    dyn::LinearGaussianLatentDynamics{T},
    step::Int,
    state::AbstractArray{T},
    extra::Nothing
) where T
    state .= state .+ dyn.σ * randn!(state)
end

function logdensity!(
    obs::LinearGaussianObservationProcess{T},
    arr::AbstractArray{T},
    timestep::Int,
    state::AbstractArray{T},
    observation::T,
    extra::Nothing
) where T <: Real
    arr .+= normlogpdf.(state, (obs.σ,), (observation,))
end

# Simulation / Inference
Tn = 100
seed = 1
N = 1_000
rng = MersenneTwister(seed)

# Float32 required for GPU but leads to numerical instability in the resampling algorithm
T = Float32
dyn = LinearGaussianLatentDynamics{T}(T(0.2))
obs = LinearGaussianObservationProcess{T}(T(0.7))
model = StateSpaceModel(dyn, obs)
xs, ys = sample(rng, model, Tn)

logevidence = filter(rng, model, N, ys)
