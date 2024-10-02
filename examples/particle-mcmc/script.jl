using AdvancedMH
using CairoMakie
using StatsBase: weights, mean

include("particles.jl")
include("resamplers.jl")
include("simple-filters.jl")

## FILTERING DEMONSTRATION #################################################################

# use a local level trend model
function simulation_model(σx²::T, σy²::T) where {T<:Real}
    init = Gaussian(zeros(T, 2), PDMat(diagm(ones(T, 2))))
    dyn = LinearGaussianLatentDynamics(T[1 1; 0 1], T[0; 0], [σx² 0; 0 0], init)
    obs = LinearGaussianObservationProcess(T[1 0], [σy²;;])
    return StateSpaceModel(dyn, obs)
end

true_params = randexp(Float32, 2);
true_model = simulation_model(true_params...);

# simulate data
rng = MersenneTwister(1234);
_, _, data = sample(rng, true_model, 150);

# test the adaptive resampling procedure
states, llbf = sample(rng, true_model, data, BF(2048, 0.5); store_ancestry=true);

# plot the smoothed states to validate the algorithm
smoothed_trend = begin
    fig = Figure(; size=(1200, 400))
    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])

    # this is gross but it works fro visualization purposes
    all_paths = map(x -> hcat(x...), get_ancestry(states.tree))
    mean_paths = mean(all_paths, weights(softmax(states.log_weights)))
    n_paths = length(all_paths)

    # plot smoothed states in black and observed data in red
    lines!(ax1, mean_paths[1, :]; color=:black)
    lines!(ax1, vcat(0, data...); color=:red, linestyle=:dash)

    # plot ancestry tree in graded black and data in red
    lines!.(ax2, getindex.(all_paths, 1, :), color=(:black, maximum([2 / n_paths, 1e-2])))
    lines!(ax2, vcat(0, data...); color=:red, linestyle=:dash)

    fig
end

## PARTICLE MCMC ###########################################################################

# consider a default Gamma prior with Float32s
prior_dist = product_distribution(Gamma(1.0f0), Gamma(1.0f0));

# basic RWMH ala AdvancedMH
function density(θ::Vector{T}) where {T<:Real}
    if insupport(prior_dist, θ)
        # _, ll = sample(rng, simulation_model(θ...), data, BF(512))
        _, ll = sample(rng, simulation_model(θ...), data, KF())
        return ll + logpdf(prior_dist, θ)
    else
        return -Inf
    end
end

pmmh = RWMH(MvNormal(zeros(Float32, 2), (0.01f0) * I));
model = DensityModel(density);

# works with AdvancedMH out of the box
chains = sample(model, pmmh, 50_000);
burn_in = 1_000;

# plot the posteriors
hist_plots = begin
    param_post = hcat(getproperty.(chains[burn_in:end], :params)...)
    fig = Figure(; size=(1200, 400))

    for i in 1:2
        # plot the posteriors with burn-in
        hist(
            fig[1, i],
            param_post[i, :];
            color=(:black, 0.4),
            strokewidth=1,
            normalization=:pdf,
        )

        # plot the true values
        vlines!(fig[1, i], true_params[i]; color=:red, linestyle=:dash, linewidth=3)
    end

    fig
end

# this is useful for SMC algorithms like SMC² or density tempered SMC
acc_ratio = mean(getproperty.(chains, :accepted))
