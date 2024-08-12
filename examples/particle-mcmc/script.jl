using AdvancedMH
using CairoMakie

include("simple-filters.jl")

true_params, simulation_model = let T = Float32
    θ = randexp(T, 3)
    dyn = LinearGaussianLatentDynamics(T[1 1;0 1], diagm(θ[1:2]))
    obs = LinearGaussianObservationProcess(T[0.5 0.5], diagm(θ[3:end]))
    θ, StateSpaceModel(dyn, obs)
end

# simulate data
rng = MersenneTwister(1234)
_, data = sample(rng, simulation_model, 150)

# consider a default Gamma prior with Float32s
prior_dist = product_distribution(Gamma(1f0), Gamma(1f0), Gamma(1f0))

#=
    Not crazy about this structure, especially since the RNG is referenced on
    the global scope. I think we can make a PMCMC sampler type which includes
    the filter algorithm within the sampler definition.

    Another issue is that we lose information on the states. Granted, this is
    also by design since that would cost a considerable amount of memory, but
    is useful nonetheless. This also needs to interface with bundle_samples()
    different than ususal, since we have the parameter space and the filtered
    states.
=#
function density(θ::Vector{T}) where {T<:Real}
    if insupport(prior_dist, θ)
        dyn = LinearGaussianLatentDynamics(T[1 1;0 1], diagm(θ[1:2]))
        obs = LinearGaussianObservationProcess(T[0.5 0.5], diagm(θ[3:end]))
        
        # _, ll = sample(rng, StateSpaceModel(dyn, obs), data, BF(512))
        _, ll = sample(rng, StateSpaceModel(dyn, obs), data, KF())
        return ll + logpdf(prior_dist, θ)
    else
        return -Inf
    end
end

# plug it into the DensityModel interface for now
pmmh  = RWMH(MvNormal(zeros(Float32, 3), (0.01f0)*I))
model = DensityModel(density)

# works with AdvancedMH out of the box
chains = sample(model, pmmh, 25_000)
burn_in = 10_000

# plot the posteriors
hist_plots = begin
    param_post = hcat(getproperty.(chains[burn_in:end], :params)...)
    fig = Figure(size = (1200, 400))

    for i in 1:3
        # plot the posteriors with burn-in
        hist(
            fig[1, i],
            param_post[i, :],
            color = :gray,
            strokewidth = 1,
            normalization = :pdf
        )

        # plot the true values
        vlines!(
            fig[1, i],
            true_params[i],
            color = :red,
            linestyle = :dash,
            linewidth = 2
        )
    end

    fig
end

# this is useful for SMC algorithms like SMC² or density tempered SMC
acc_ratio = mean(getproperty.(chains, :accepted))