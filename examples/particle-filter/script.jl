# # Partilce Filter with adaptive resampling
using StatsBase
using AbstractMCMC
using Random
using SSMProblems
using Distributions
using Plots
using StatsFuns

# Particle Filter 
ess(weights) = inv(sum(abs2, weights))
get_weights(logweights::T) where {T<:AbstractVector{<:Real}} = StatsFuns.softmax(logweights)

function systematic_resampling(
    rng::AbstractRNG, weights::AbstractVector{<:Real}, n::Integer=length(weights)
)
    return rand(rng, Distributions.Categorical(weights), n)
end

function sweep!(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    particles,
    observations::AbstractArray,
    resampling=systematic_resampling,
    threshold=0.5,
)
    N = length(particles[:, 1])
    logweights = zeros(N)

    for (timestep, observation) in enumerate(observations)
        weights = get_weights(logweights)
        if ess(weights) <= threshold * N
            idx = resampling(rng, weights)
            particles = particles[idx, timestep]
            fill!(logweights, 0)
        end

        latent_state = zeros(N)
        for i in eachindex(particles[:, timestep])
            latent_state[i]= SSMProblems.simulate(rng, model.dyn, timestep, particles[i, timestep], nothing)
            logweights[i] += SSMProblems.logdensity(
                model.obs, timestep, particles[i, timestep], observation, nothing
            )
        end
        particles[:, timestep] = latent_state
    end

    idx = resampling(rng, get_weights(logweights))
    return particles[idx, end]
end

# Turing style sample method
function StatsBase.sample(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    n::Int,
    observations::AbstractVector;
    resampling=systematic_resampling,
    threshold=0.5,
)
    T = length(observations)
    particles = zeros(Float64, n, T)
    vec = map(1:N) do i
        state = SSMProblems.simulate(rng, model.dyn, nothing)
    end
    particles[:, 1] = collect(vec)
    samples = sweep!(rng, model, particles, observations, resampling, threshold)
    return particles
end

# Inference code
struct LinearGaussianLatentDynamics{T<:Real} <: LatentDynamics
    σ::T
end

struct LinearGaussianObservationProcess{T<:Real} <: ObservationProcess
    σ::T
end

const LinearGaussianSSM{T} = StateSpaceModel{
    <:LinearGaussianLatentDynamics{T},<:LinearGaussianObservationProcess{T}
};

function SSMProblems.distribution(dyn::LinearGaussianLatentDynamics, extra::Nothing)
    return Normal(0, dyn.σ)
end

function SSMProblems.distribution(dyn::LinearGaussianLatentDynamics, step::Int, state::Real, extra::Nothing)
    return Normal(state, dyn.σ)
end

function SSMProblems.distribution(obs::LinearGaussianObservationProcess, step::Int, state::Real, extra::Nothing)
    return Normal(state, dyn.σ)
end


# Simulation
T = 150
seed = 1
N = 500
rng = MersenneTwister(seed)

dyn = LinearGaussianLatentDynamics(0.2)
obs = LinearGaussianObservationProcess(0.7)
model = StateSpaceModel(dyn, obs)
xs, ys = sample(rng, model, T)

particles = sample(rng, model, N, ys)

plot(xs; label="True state", linewidth=2)
plot(ys; label="True state", linewidth=2)
gui()

a = 1
