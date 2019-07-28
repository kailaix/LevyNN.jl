using ADCME
using PyCall
using LinearAlgebra
using PyPlot
using Random
Random.seed!(233)

if Sys.islinux()
py"""
import tensorflow as tf
libRBFPoly = tf.load_op_library('build/libRBFPoly.so')
@tf.custom_gradient
def rbf_poly(x,theta,c,p,ac,h):
    y = libRBFPoly.rbf_poly(x,theta,c,p,ac,h)
    def grad(dy):
        return libRBFPoly.rbf_poly_grad(dy, y, x,theta,c,p,ac,h)
    return y, grad
"""
elseif Sys.isapple()
py"""
import tensorflow as tf
libRBFPoly = tf.load_op_library('build/libRBFPoly.dylib')
@tf.custom_gradient
def rbf_poly(x,theta,c,p,ac,h):
    y = libRBFPoly.rbf_poly(x,theta,c,p,ac,h)
    def grad(dy):
        return libRBFPoly.rbf_poly_grad(dy, y, x,theta,c,p,ac,h)
    return y, grad
"""
elseif Sys.iswindows()
py"""
import tensorflow as tf
libRBFPoly = tf.load_op_library('build/libRBFPoly.dll')
@tf.custom_gradient
def rbf_poly(x,theta,c,p,ac,h):
    y = libRBFPoly.rbf_poly(x,theta,c,p,ac,h)
    def grad(dy):
        return libRBFPoly.rbf_poly_grad(dy, y, x,theta,c,p,ac,h)
    return y, grad
"""
end

rbf_poly = py"rbf_poly"

# TODO: 
c = constant(1.0)
ac = constant([0.0;0.0])
h = constant(0.01)
theta = constant(ones(100,100))
x0 = rand(1000,2)
x = constant(x0)
p = constant(rand(6))
u = rbf_poly(x,theta,c,p,ac,h)
sess = Session()
init(sess)
run(sess, u)
# error("")
# TODO: 


# gradient check -- v
function scalar_function(m)
    # return sum(rbf_poly(x,theta,m,p,ac,h))
    return sum(rbf_poly(x,theta,c,m,ac,h))
end

# x = constant([1.0 1.0])
m_ = constant(1.0)
v_ = rand()
m_ = constant(rand(6))
v_ = rand(6)

y_ = scalar_function(m_)
dy_ = gradients(y_, m_)
ms_ = Array{Any}(undef, 5)
ys_ = Array{Any}(undef, 5)
s_ = Array{Any}(undef, 5)
w_ = Array{Any}(undef, 5)
gs_ =  @. 1 / 10^(1:5)
gs_ = gs_/10^4

for i = 1:5
    g_ = gs_[i]
    ms_[i] = m_ + g_*v_
    ys_[i] = scalar_function(ms_[i])
    s_[i] = ys_[i] - y_
    w_[i] = s_[i] - g_*sum(v_*dy_)
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
