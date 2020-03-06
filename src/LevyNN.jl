module LevyNN
using PyCall
using Distributions
using Statistics
using ADCME
using LinearAlgebra
using PyPlot
np = PyNULL()
quadpy = PyNULL()

function __init__()
    global np, quadpy
    copy!(np, pyimport("numpy"))
    copy!(quadpy, pyimport("quadpy"))
    global ploned = load_op_and_grad("$(@__DIR__)/../deps/PL1D/build/libPLONED", "ploned")
    global rbfoned = load_op_and_grad("$(@__DIR__)/../deps/RBF1D/build/libRBFONED", "rbfoned")
end
include("CustomDistributions.jl")
include("Optim.jl")
include("Ops.jl")
include("Quadrature.jl")
include("BasisFunctions.jl")
include("CF.jl")
include("Simulation.jl")
include("Utils.jl")
end