using Distributions
using FredData
using CairoMakie

include("particles.jl")
include("resamplers.jl")
include("simple-filters.jl")

## LATENT DYNAMICS #########################################################################

# this just encompasses common routines between OutlierAdjustedTrend{T} and SimpleTrend{T}
abstract type LocalLevelTrend{T} <: LatentDynamics{Vector{T}} end

function SSMProblems.logdensity(
    proc::LocalLevelTrend{T}, step::Integer, prev_state::Vector{T}, state::Vector{T}, extra
) where {T<:Real}
    vol_prob = logpdf(MvNormal(prev_state[2:end], proc.γ), state[2:end])
    trend_prob = logpdf(Normal(prev_state[1], 0.5 * exp(prev_state[2])), state[1])
    return vol_prob + trend_prob
end

struct OutlierAdjustedTrend{T} <: LocalLevelTrend{T}
    γ::Vector{T}
    switch_dist::Bernoulli{T}
    outlier_dist::Uniform{T}
end

function SSMProblems.distribution(proc::OutlierAdjustedTrend{T}, extra) where {T<:Real}
    return product_distribution(
        Normal(zero(T), T(1)), Normal(zero(T), T(1)), Normal(zero(T), T(5)), Dirac(one(T))
    )
end

function SSMProblems.simulate(
    rng::AbstractRNG, proc::OutlierAdjustedTrend{T}, step::Integer, state::Vector{T}, extra
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[2:3] += proc.γ .* randn(rng, T, 2)
    new_state[1] += 0.5 * exp(new_state[2]) * randn(T)
    new_state[4] = rand(rng, proc.switch_dist) ? rand(rng, proc.outlier_dist) : one(T)
    return new_state
end

struct SimpleTrend{T} <: LocalLevelTrend{T}
    γ::Vector{T}
end

function SSMProblems.distribution(proc::SimpleTrend{T}, extra) where {T<:Real}
    return product_distribution(
        Normal(zero(T), T(1)), Normal(zero(T), T(1)), Normal(zero(T), T(5))
    )
end

function SSMProblems.simulate(
    rng::AbstractRNG, proc::SimpleTrend{T}, step::Integer, state::Vector{T}, extra
) where {T<:Real}
    new_state = deepcopy(state)
    new_state[2:3] += proc.γ .* randn(rng, T, 2)
    new_state[1] += 0.5 * exp(new_state[2]) * randn(T)
    return new_state
end

## OBSERVATION PROCESS #####################################################################

struct OutlierAdjustedObservation{T} <: ObservationProcess{T} end

function SSMProblems.distribution(
    proc::OutlierAdjustedObservation{T}, step::Integer, state::Vector{T}, extra
) where {T<:Real}
    return Normal(state[1], sqrt(state[4]) * 0.5 * exp(state[3]))
end

struct SimpleObservation{T} <: ObservationProcess{T} end

function SSMProblems.distribution(
    proc::SimpleObservation{T}, step::Integer, state::Vector{T}, extra
) where {T<:Real}
    return Normal(state[1], 0.5 * exp(state[3]))
end

## TESTING #################################################################################

# include UCSV as a baseline
function UCSV(γ::T) where {T}
    dyn = SimpleTrend(fill(γ, 2))
    obs = SimpleObservation{T}()
    return StateSpaceModel(dyn, obs)
end

# quick demo of the outlier-adjusted univariate UCSV model
function UCSVO(γ::T, prob::T) where {T}
    dyn = OutlierAdjustedTrend(fill(γ, 2), Bernoulli(prob), Uniform(T(2), T(10)))
    obs = OutlierAdjustedObservation{T}()
    return StateSpaceModel(dyn, obs)
end

# wrapper to plot and demo the model
function plot_ucsv(rng::AbstractRNG, model, data)
    filter = BF(2^12; threshold=1.0, resampler=Systematic())
    sparse_ancestry = AncestorCallback(eltype(model.dyn), filter.N, 1.0)
    states, ll = sample(rng, model, filter, data; callback=sparse_ancestry)

    fig = Figure(; size=(800, 300))

    all_paths = map(x -> hcat(x...), get_ancestry(sparse_ancestry.tree))
    mean_paths = mean(all_paths, weights(softmax(states.log_weights)))

    ax = Axis(fig[1:2, 1]; limits=(nothing, (-14, 18)), title="Trend Inflation")
    lines!(fig[1:2, 1], vcat(0, data...); color=:red, linestyle=:dash)
    lines!(ax, mean_paths[1, :]; color=:black)

    ax1 = Axis(fig[1, 2]; title="Volatility")
    ax2 = Axis(fig[2, 2])
    lines!(ax1, 0.5 * exp.(mean_paths[2, :]); color=:black)
    lines!(ax2, 0.5 * exp.(mean_paths[3, :]); color=:black)

    display(fig)
    return ll
end

# query inflation data from FRED
fred = get_data(
    Fred(),
    "PCEPI";
    observation_start="1959-06-01",
    observation_end="2016-01-01",
    units="pca",
    frequency="q",
    aggregation_method="eop",
);

rng = MersenneTwister(1234);

# plot both models side by side, notice the difference in volatility
plot_ucsv(rng, UCSV(0.2), fred.data.value);
plot_ucsv(rng, UCSVO(0.2, 0.1), fred.data.value);
