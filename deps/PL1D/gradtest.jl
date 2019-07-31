using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libPLONED = tf.load_op_library('build/libPLONED.so')
@tf.custom_gradient
def ploned(x,u):
    y = libPLONED.ploned(x,u)
    def grad(dy):
        return libPLONED.ploned_grad(dy, y, x,u)
    return y, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libPLONED = tf.load_op_library('build/libPLONED.dylib')
@tf.custom_gradient
def ploned(x,u):
    y = libPLONED.ploned(x,u)
    def grad(dy):
        return libPLONED.ploned_grad(dy, y, x,u)
    return y, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libPLONED = tf.load_op_library('build/libPLONED.dll')
@tf.custom_gradient
def ploned(x,u):
    y = libPLONED.ploned(x,u)
    def grad(dy):
        return libPLONED.ploned_grad(dy, y, x,u)
    return y, grad
"""
end

ploned = py"ploned"

# TODO: 
u = sin.(LinRange(0,2π,10)[1:9])
x = constant(LinRange(0,2π,1000))
u = ploned(x,u)
sess = Session()
init(sess)
run(sess, u)
# error()
# TODO: 


# gradient check -- v
function scalar_function(m)
    return sum(tanh(ploned(x,m)))
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
