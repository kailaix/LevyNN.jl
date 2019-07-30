export UniformDisk, MixedGaussian
struct UniformDisk
    r::Float64
end
function Base.:rand(uf::UniformDisk)
    # r = uf.r
    # local x, y
    # ac = false
    # while !ac 
    #     x = (2*rand()-1)*r 
    #     y = (2*rand()-1)*r 
    #     if x^2 + y^2 <= r^2 && x>0
    #         ac = true
    #     end
    # end
    # return [x; y]
    θ = randn(2)
    θ /= norm(θ)
    θ[1] = abs(θ[1])
    return θ
    # d = Exponential()
    # r = rand(d)
    # θ = rand()*2π
    # x = r*cos(θ); y = r*sin(θ)
    # x = abs(x)
    
end

struct MixedGaussian
end
function Base.:rand(mg::MixedGaussian)
    if rand()>0.75
        return randn(2) .+ 1.0
    else
        return randn(2) .- 1.0
    end
end