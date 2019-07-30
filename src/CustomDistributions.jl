export TruncatedNormal2D, MixedGaussian2D, StandardNormal2D, Uniform1D, Step1D
struct TruncatedNormal2D end
function Base.:rand(uf::TruncatedNormal2D)
    θ = randn(2)
    θ /= norm(θ)
    θ[1] = abs(θ[1])
    return θ
end

struct MixedGaussian2D end
function Base.:rand(mg::MixedGaussian2D)
    if rand()>0.75
        return randn(2) .+ 1.0
    else
        return randn(2) .- 1.0
    end
end

struct StandardNormal2D end
function Base.:rand(mg::StandardNormal2D)
    randn(2)
end

struct Uniform1D end
function Base.:rand(uf::Uniform1D)
    s = randn(2)
    s /= norm(s)
    s
end

struct Step1D end
function Base.:rand(uf::Step1D)
    local s 
    while true
        s = randn(2)
        s /= norm(s)
        if abs(s[1])>0.5
            break
        end
    end
    return s
end

