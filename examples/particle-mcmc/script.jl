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

# generate model and data
rng = MersenneTwister(1234)
true_params = randexp(rng, Float32, 2);
true_model = simulation_model(true_params...);
_, _, data = sample(rng, true_model, 150);

# run a bootstrap filter, resampling at every iteration with a Rejection resampler
filter = BF(1024; threshold=1.0, resampler=Rejection());
sparse_ancestry = begin
    M = floor(filter.N * log(filter.N))
    states = initialise(rng, true_model, filter, nothing)
    sparse_ancestry = ParticleTree(states.filtered, Int64(M))

    for t in eachindex(data)
        proposed_states = predict(rng, true_model, filter, t, states, nothing)
        states, log_marginal = update(
            true_model, filter, t, proposed_states, data[t], nothing
        )

        prune!(sparse_ancestry, get_offspring(states.ancestors))
        insert!(sparse_ancestry, states.filtered, states.ancestors)
    end

    sparse_ancestry
end;

smoothed_trend = try
    fig = Figure(; size=(600, 400))
    ax1 = Axis(fig[1, 1])

    # this is gross but it works fro visualization purposes
    all_paths = map(x -> hcat(x...), get_ancestry(sparse_ancestry))
    n_paths = length(all_paths)

    # plot ancestry tree in graded black and data in red
    lines!.(ax1, getindex.(all_paths, 1, :), color=(:black, maximum([2 / n_paths, 1e-2])))
    lines!(ax1, vcat(0, data...); color=:red, linestyle=:dash)

    fig
catch
    # keep this here until the callbacks are in a stable enough 
    @error "Sparse ancestry storage callbacks not yet implemented, this will error"
end

## PARTICLE MCMC ###########################################################################

# consider a default Gamma prior with Float32s
prior_dist = product_distribution(Gamma(1.0f0), Gamma(1.0f0));

# basic RWMH ala AdvancedMH
function density(θ::Vector{T}) where {T<:Real}
    if insupport(prior_dist, θ)
        # _, ll = sample(rng, simulation_model(θ...), BF(512), data)
        _, ll = sample(rng, simulation_model(θ...), KF(), data)
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
