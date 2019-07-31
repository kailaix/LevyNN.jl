export RBF, RBF1D, PL, PL1D, NN, Delta, evaluate, evaluate1D



mutable struct RBF
    c::PyObject
    ac::PyObject
    h::PyObject
    θ::PyObject
    p::PyObject
    nparams::Int64
end

# RBF approximation within [-R,R]x[-R,R], and each side has n centers
function RBF(R::Float64, n::Int64, c::Float64=0.0; poly::Bool = true)
    if c==0.0
        c = 2R/n
    end
    c = constant(c)
    ac = constant([-R;-R])
    h = constant(2R/n)
    nparams = 0.0
    θ = Variable(zeros(n+1, n+1))
    nparams += (n+1)^2
    if poly
        nparams += 3
        p = [Variable(zeros(3));constant(zeros(3))]
    else
        p = constant(zeros(6))
    end
    RBF(c, ac, h, θ, p, nparams)
end

function evaluate(rbf::RBF, x::Union{PyObject, Array{Float64}})
    if isa(x, Array)
        x = constant(x)
    end
    rbf_poly(x,rbf.θ,rbf.c,rbf.p,rbf.ac,rbf.h)
end

mutable struct PL
    ac::PyObject
    h::PyObject
    θ::PyObject
    nparams::Int64
end

# PL approximation within [-R,R]x[-R,R], and each side has n centers
function PL(R::Float64, n::Int64)
    ac = constant([-R;-R])
    h = constant(2R/n)
    θ = Variable(zeros(n+1, n+1))
    PL(ac, h, θ, (n+1)^2)
end


function evaluate(plfun::PL, x::Union{PyObject, Array{Float64}})
    if isa(x, Array)
        x = constant(x)
    end
    pl(x,plfun.θ,plfun.ac,plfun.h)
end

function symmetrize(x)
    y0 = zeros(size(x,1))
    x0 = x
    for i = 1:length(y0)
        y0[i] = atan(x0[i,2],x0[i,1])
        y0[i] += π
    end
    y0
end

struct PL1D
    θ::PyObject
    nparams::Int64
end

function PL1D(n::Int64)
    θ = Variable(zeros(n))
    PL1D(θ, n)
end

function evaluate1D(plfun::PL1D, x::Array{Float64})
    x0 = symmetrize(x)
    x0 = constant(x0)
    ploned(x0,plfun.θ)+ploned(2π-x0,plfun.θ)
end



struct RBF1D
    θ::PyObject
    c::PyObject
    nparams::Int64
end

function RBF1D(n::Int64)
    c = 2π/n
    θ = Variable(zeros(n))
    RBF1D(θ, constant(c), n)
end

function evaluate1D(plfun::RBF1D, x::Array{Float64})
    x0 = symmetrize(x)
    x0 = constant(x0)
    rbfoned(x0,plfun.θ, plfun.c)+rbfoned(2π-x0,plfun.θ, plfun.c)
end


mutable struct NN
    config::Array{Int64}
    scope::String
    nparams::Int64
end

function NN(config::Array{Int64}, scope::String)
    l = [2;config]
    nparams = 0.0
    for i = 1:length(l)-1
        nparams += l[i]*l[i+1] + l[i+1]
    end
    NN(config, scope, nparams)
end

function evaluate(nn::NN, x::Union{PyObject, Array{Float64}})
    scope = nn.scope
    output_dims = nn.config
    if isa(x, Array)
        x = constant(x)
    end

    flag = false
    if length(size(x))==1
        x = reshape(x, length(x), 1)
        flag = true
    end
    net = x
    variable_scope(scope, reuse=AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) do
        for i = 1:length(output_dims)-1
            net = dense(net, output_dims[i], activation="relu")
        end
        net = dense(net, output_dims[end])
    end
    if flag && (size(net,2)==1)
        net = squeeze(net)
    end
    
    # squeeze(ae(x, nn.config, nn.scope))
    squeeze((net))
end

function evaluate1D(nn::NN, x::Union{PyObject, Array{Float64}})
    evaluate(nn, x) + evaluate(nn, -x)
end

struct Delta
    y::PyObject
end

function Delta(n::Int64)
    x = Variable(zeros(n))
    Delta(x)
end

function evaluate(delta::Delta, x::Union{PyObject, Array{Float64}})
    delta.y
end



