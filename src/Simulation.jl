export LevySimulator, StableSimulator, simulate, evaluateECF

function Base.:rand(n::Nothing)
    return zeros(2)
end

mutable struct LevySimulator
    Δt::Float64
    A::Array{Float64}
    b::Array{Float64}
    λ::Float64
    Jump::Distribution
    Nv::Union{Distribution, Nothing}
end

function LevySimulator(A::Array{Float64}, b::Array{Float64}, λ::Float64, Jump::Distribution, Δt::Float64)
    local Nv
    if minimum(eigvals(A))<=0.0 || !(norm(A-A')≈0.0)
        Nv = nothing
    else
        Nv = MvNormal(b*Δt, A*Δt)
    end
    LevySimulator(Δt, A, b, λ, Jump, Nv)
end

struct Stable
    σ::Float64
    α::Float64
    exponetial::Distribution
end

function Base.:rand(s::Stable, n::Int64)
    γ = (rand(n)-0.5)*π
    w = rand(s.exponetial, n)
    @. σ^(1/α) * sin(α*γ)/(cos(γ)^(1/α)) * (cos((1-α)*γ)/w)^((1-α)/α)
end

function Base.:rand(s::Stable)
    γ = (rand()-0.5)*π
    w = rand(s.exponetial)
    σ^(1/α) * sin(α*γ)/(cos(γ)^(1/α)) * (cos((1-α)*γ)/w)^((1-α)/α)
end

mutable struct StableSimulator
    Δt::Float64
    A::Array{Float64}
    b::Array{Float64}
    α::Float64
    λ::Float64
    αStable::Stable
    Direction::Distribution
    Nv::Distribution
end


function StableSimulator(A::Array{Float64}, b::Array{Float64}, α::Float64, λ::Float64, Direction::Distribution, Δt::Float64)
    Nv = MvNormal(b*Δt, A*Δt)
    StableSimulator(Δt, A, b, α, λ, Stable(λ*Δt, α, Exponential()), Direction, Nv)
end

function simulate(ls::LevySimulator, x0::Array{Float64}, n::Int64)
    # Reference: https://quant.stackexchange.com/questions/29606/how-to-simulate-a-jump-diffusion-process
    local P, Nt
    λ = ls.λ
    X = zeros(n,2)
    if λ!=0.0
        P = Poisson(λ*ls.Δt)
    end
    X[1,:] = x0
    for t = 2:n 
        J = zeros(2)
        if λ!=0.0
            Nt = rand(P)
            for i = 1:Nt
                J += rand(ls.Jump)
            end
        end
        J += rand(ls.Nv)
        X[t,:] = X[t-1,:] + J
    end
    ΔX = diff(X, dims=1)
    X, ΔX
end

function simulate(ss::StableSimulator, x0::Array{Float64}, n::Int64)
    local P, Nt
    λ = ls.λ
    X = zeros(n,2)
    X[1,:] = x0
    for t = 2:n 
        J = zeros(2)
        if λ!=0.0
            J += rand(ss.αStable)*rand(ss.Direction)
        end
        J += rand(ls.Nv)
        X[t,:] = X[t-1,:] + J
    end
    X
end

function evaluateECF(X::Array{Float64}, ξ::Array{Float64})
    E = zeros(ComplexF64, size(ξ,1))
    for i = 1:size(ξ,1)
        E[i] = mean(exp.(1.0im * X * ξ[i,:]))
    end
    E
end

function evaluateECF(X::Array{Float64}, ξ::PyObject)
    function _evaluate(ξ)
        mean(exp(1.0im * X * ξ))
    end
    tf.map_fn(_evaluate, ξ, dtype=tf.complex128)
end

