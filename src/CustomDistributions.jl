export TruncatedNormal2D, MixedGaussian2D, TruncatedUniform2D, StandardNormal2D, Γstep, Γuniform, Γx2
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
