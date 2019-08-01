export TruncatedNormal2D, MixedGaussian2D, TruncatedUniform2D, StandardNormal2D, Γstep, Γuniform, Γx2, pdf
struct TruncatedNormal2D end
function Base.:rand(uf::TruncatedNormal2D)
    θ = randn(2)
    θ /= norm(θ)
    θ[1] = abs(θ[1])
    return θ
end

struct TruncatedUniform2D end
function Base.:rand(uf::TruncatedUniform2D)
    local θ
    while true
        θ = randn(2)
        if norm(θ)>1 || θ[1]<0 || θ[2]<0
            continue
        else
            break
        end
    end
    return θ
end

function Distributions.:pdf(uf::TruncatedNormal2D, x::Array{Float64})
    y = zeros(size(x,1))
    for i = 1:length(y)
        if norm(x[i,:])>1 || x[i,1]<0 || x[i,2]<0
            continue
        end
        y[i] = 2/π/(1-exp(-0.5))*exp(-(norm(x[i,:])^2)/2)
    end
    y
end

struct MixedGaussian2D end
function Base.:rand(mg::MixedGaussian2D)
    if rand()>0.75
        return randn(2) .+ 1.0
    else
        return randn(2) .- 1.0
    end
end

function Distributions.:pdf(uf::MixedGaussian2D, x::Array{Float64})
    y = zeros(size(x,1))
    for i = 1:length(y)
        y[i] = 1/(2π)*exp(-(norm(x[i,:])^2)/2)*0.5 + 1/(2π)*exp(-(norm(x[i,:]-[1.0;1.0])^2)/2)*0.5
    end
    y
end

struct StandardNormal2D end
function Base.:rand(mg::StandardNormal2D)
    randn(2)
end

function Distributions.:pdf(uf::StandardNormal2D, x::Array{Float64})
    y = zeros(size(x,1))
    for i = 1:length(y)
        y[i] = 1/(2π)*exp(-(norm(x[i,:])^2)/2)
    end
    y
end

function Γuniform(x)
    return ones(size(x,1))
end

function Γstep(x)
    function helper(x)
        if abs(x[1])>0.5
            return 1.0
        else
            return 0.0
        end
    end
    y = zeros(size(x,1))
    for i = 1:size(x,1)
        y[i] = helper(x[i,:])
    end
    y
end

function Γx2(x)
    function helper(x)
        5x[1]^2+x[2]^2
    end
    y = zeros(size(x,1))
    for i = 1:size(x,1)
        y[i] = helper(x[i,:])
    end
    y
end
