export LevySimulator, StableSimulator, simulate, evaluateECF, SpectralMeasureSimulator

function Base.:rand(n::Nothing)
    return zeros(2)
end


struct SpectralMeasureSampler
    y::Array{Float64}
    α::Float64
    s::Array{Float64}
end

# simulate exp(-int |ξ'*s|^α * Γ(ds))
function SpectralMeasureSampler(Γ::Function, α::Float64, n::Int64=50)
    q = Quadrature1D(n)
    s = q.points
    y = Γ(s).*q.weights
    # println("creating a new sms")
    SpectralMeasureSampler(y, α, s)
end




function Base.:rand(sms::SpectralMeasureSampler)
    simulate_discrete_measure(sms.y, sms.α, sms.s)
end

function simulate_discrete_measure(y, α, s)
    n = length(y)
    γ = (rand(n) .- 0.5)*π
    w = rand(Exponential(), n)
    z = @. sin(α*γ)/(cos(γ)^(1/α)) * (cos((1-α)*γ)/w)^((1-α)/α) 
    z = @. z*y^(1/α)
    sum([z z].*s, dims=1)[:]
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


function simulate(ls::LevySimulator, x0::Array{Float64}, n::Int64)
    # Reference: https://quant.stackexchange.com/questions/29606/how-to-simulate-a-jump-diffusion-process
    local P
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


mutable struct StableSimulator
    Δt::Float64
    A::Array{Float64}
    b::Array{Float64}
    α::Float64
    λ::Float64
    sms::SpectralMeasureSampler
    Nv::Union{Distribution, Nothing}
end


function StableSimulator(A::Array{Float64}, b::Array{Float64}, α::Float64, λ::Float64, Γ::Function, Δt::Float64, sms_order::Int64=50)
    local Nv
    if minimum(eigvals(A))<=0.0 || !(norm(A-A')≈0.0)
        Nv = nothing
    else
        Nv = MvNormal(b*Δt, A*Δt)
    end
    Γnew = x->λ*Δt*Γ(x)
    sms = SpectralMeasureSampler(Γnew, α, sms_order)
    StableSimulator(Δt, A, b, α, λ, sms, Nv)
end

function simulate(ss::StableSimulator, x0::Array{Float64}, n::Int64)
    λ = ss.λ
    X = zeros(n,2)
    X[1,:] = x0
    for t = 2:n 
        J = zeros(2)
        if λ!=0.0
            J += rand(ss.sms)
        end
        J += rand(ss.Nv)
        X[t,:] = X[t-1,:] + J
    end
    ΔX = diff(X, dims=1)
    X, ΔX
end

function evaluateECF(X::Array{Float64}, ξ::Array{Float64})
    E = zeros(ComplexF64, size(ξ,1))
    for i = 1:size(ξ,1)
        E[i] = mean(exp.(1.0im * X * ξ[i,:]))
    end
    E
end


