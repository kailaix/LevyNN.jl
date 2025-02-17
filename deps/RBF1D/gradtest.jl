using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libRBFONED = tf.load_op_library('build/libRBFONED.so')
@tf.custom_gradient
def rbfoned(x,u,c):
    y = libRBFONED.rbfoned(x,u,c)
    def grad(dy):
        return libRBFONED.rbfoned_grad(dy, y, x,u,c)
    return y, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libRBFONED = tf.load_op_library('build/libRBFONED.dylib')
@tf.custom_gradient
def rbfoned(x,u,c):
    y = libRBFONED.rbfoned(x,u,c)
    def grad(dy):
        return libRBFONED.rbfoned_grad(dy, y, x,u,c)
    return y, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libRBFONED = tf.load_op_library('build/libRBFONED.dll')
@tf.custom_gradient
def rbfoned(x,u,c):
    y = libRBFONED.rbfoned(x,u,c)
    def grad(dy):
        return libRBFONED.rbfoned_grad(dy, y, x,u,c)
    return y, grad
"""
end

rbfoned = py"rbfoned"

# TODO: 
u = sin.(LinRange(0,2π,10)[1:9])
x = constant(LinRange(0,2π,1000))
c = constant(2π/10)
u = rbfoned(x,u,c)
sess = Session()
init(sess)
run(sess, u)
# error()
# TODO: 

# θ = ones(10)
# c = 2π/10
# u = rbfoned(LinRange(0,2π,10000)|>Array|>constant, constant(θ), constant(c))
# close("all")
# plot(LinRange(0,2π,10000), run(sess, u))
# savefig("rbftest.png")
# error()
# gradient check -- v
function scalar_function(m)
    return sum(tanh(rbfoned(x,m,c)))
end

m_ = constant(sin.(LinRange(0,2π,10)[1:9]))
v_ = rand(9)
y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_.*dy_)
end

sess = Session()
init(sess)
sval_ = run(sess, s_)
wval_ = run(sess, w_)
close("all")
loglog(gs_, abs.(sval_), "*-", label="finite difference")
loglog(gs_, abs.(wval_), "+-", label="automatic differentiation")
loglog(gs_, gs_.^2 * 0.5*abs(wval_[1])/gs_[1]^2, "--",label="\$\\mathcal{O}(\\gamma^2)\$")
loglog(gs_, gs_ * 0.5*abs(sval_[1])/gs_[1], "--",label="\$\\mathcal{O}(\\gamma)\$")

plt.gca().invert_xaxis()
legend()
xlabel("\$\\gamma\$")
ylabel("Error")
