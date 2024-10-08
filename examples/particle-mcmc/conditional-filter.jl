using CairoMakie

include("particles.jl")
include("resamplers.jl")
include("simple-filters.jl")

## MULTICALLBACKS ##########################################################################

# borrowed from TuringCallbacks, and repurposed to double check reference trajectories exist
struct MultiCallback{Cs}
    callbacks::Cs
end

MultiCallback() = MultiCallback(())
MultiCallback(callbacks...) = MultiCallback(callbacks)

(c::MultiCallback)(args...; kwargs...) = foreach(c -> c(args...; kwargs...), c.callbacks)

## CONDITINAL SMC ##########################################################################

struct ConditionalSMC{F<:AbstractFilter} <: AbstractSampler
    filter::F
    N::Integer
end

function CSMC(filter::AbstractFilter, N::Integer)
    return ConditionalSMC(filter, N)
end

# this is pretty useless without sampling model parameters
function sample(
    rng::AbstractRNG,
    model::StateSpaceModel,
    sampler::ConditionalSMC,
    observations::AbstractVector;
    kwargs...,
)
    # not type stable, but I'm not that concerned with it right now
    star_trajectory = nothing
    multi_callback = nothing
    ll = zero(eltype(model))

    for n in 1:(sampler.N)
        # store resampling index for testing purposes
        multi_callback = MultiCallback(
            AncestorCallback(eltype(model.dyn), sampler.filter.N, 1.0),
            ResamplerCallback(sampler.filter.N),
        )

        states, ll = sample(
            rng,
            model,
            sampler.filter,
            observations;
            ref_state=star_trajectory,
            callback=multi_callback,
        )

        weights = softmax(states.log_weights)
        star_trajectory = rand(rng, multi_callback.callbacks[1].tree, weights)
        println("n = $n \t $ll")
    end

    return multi_callback.callbacks[2], ll
end

## PARTICLE GIBBS ##########################################################################

#=
    TODO: think about interfacing with AbstractMCMC for something like this
=#

## TESTING #################################################################################

# use a local level trend model
function simulation_model(σx²::T, σy²::T) where {T<:Real}
    init = Gaussian(zeros(T, 2), PDMat(diagm(ones(T, 2))))
    dyn = LinearGaussianLatentDynamics(T[1 1; 0 1], T[0; 0], [σx² 0; 0 0], init)
    obs = LinearGaussianObservationProcess(T[1 0], [σy²;;])
    return StateSpaceModel(dyn, obs)
end

# generate model and data
rng = MersenneTwister(1234);
true_params = randexp(rng, Float64, 2);
true_model = simulation_model(true_params...);
_, _, data = sample(rng, true_model, 50);

filter = BF(20; threshold=1.0, resampler=Systematic());
rs_path, ll = sample(rng, true_model, CSMC(filter, 5), data);

# check that y = 1 always has a path
rs_check = begin
    fig = Figure(; size=(600, 300))
    ax = Axis(
        fig[1, 1];
        xticks=0:10:50,
        yticks=0:10:(filter.N),
        limits=(nothing, (-5, filter.N + 5)),
    )

    paths = get_ancestry(rs_path.tree)
    scatterlines!.(
        ax, paths, color=(:black, 0.25), markercolor=:black, markersize=5, linewidth=1
    )

    fig
end
