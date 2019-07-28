using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libPL = tf.load_op_library('build/libPL.so')
@tf.custom_gradient
def pl(x,theta,ac,h):
    y = libPL.pl(x,theta,ac,h)
    def grad(dy):
        return libPL.pl_grad(dy, y, x,theta,ac,h)
    return y, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libPL = tf.load_op_library('build/libPL.dylib')
@tf.custom_gradient
def pl(x,theta,ac,h):
    y = libPL.pl(x,theta,ac,h)
    def grad(dy):
        return libPL.pl_grad(dy, y, x,theta,ac,h)
    return y, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libPL = tf.load_op_library('build/libPL.dll')
@tf.custom_gradient
def pl(x,theta,ac,h):
    y = libPL.pl(x,theta,ac,h)
    def grad(dy):
        return libPL.pl_grad(dy, y, x,theta,ac,h)
    return y, grad
"""
end

pl = py"pl"

# TODO: 
ac = constant([0.0;0.0])
h = constant(0.01)
theta = constant(ones(101,101))
x0 = rand(1000,2)
x = constant(x0)

u = pl(x,theta,ac,h)
sess = Session()
init(sess)
uval = run(sess, u)
# scatter3D(x0[:,1], x0[:,2], uval)
# error("")

# TODO: 


# gradient check -- v
function scalar_function(m)
    return sum(tanh(pl(x,m,ac,h)))
end

m_ = constant(rand(101,101))
v_ = rand(101,101)
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
