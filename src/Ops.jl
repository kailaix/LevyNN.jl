function quadgauss(n::Int64, x::Float64, y::Float64, R::Float64)
    xs = zeros(n^2); ys = zeros(n^2); ws=zeros(n^2)
    if Sys.isapple()
        ccall((:quadgauss_sphere, "./deps/quadrature/build/libgauss.dylib"), Cvoid,
        (Cint, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}), n, R, x, y, xs, ys, ws);
    elseif Sys.islinux()
        ccall((:quadgauss_sphere, "./deps/quadrature/build/libgauss.so"), Cvoid,
        (Cint, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}), n, R, x, y, xs, ys, ws);
    elseif Sys.iswindows()
        ccall((:quadgauss_sphere, "./deps/quadrature/libgauss.ddl"), Cvoid,
        (Cint, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}), n, R, x, y, xs, ys, ws);
    end
    return [xs ys], ws
end


if Sys.islinux()
py"""
import tensorflow as tf
libRBFPoly = tf.load_op_library("./deps/RBF/build/libRBFPoly.so")
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
libRBFPoly = tf.load_op_library("./deps/RBF/build/libRBFPoly.dylib")
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
libRBFPoly = tf.load_op_library("./deps/RBF/build/libRBFPoly.dll")
@tf.custom_gradient
def rbf_poly(x,theta,c,p,ac,h):
    y = libRBFPoly.rbf_poly(x,theta,c,p,ac,h)
    def grad(dy):
        return libRBFPoly.rbf_poly_grad(dy, y, x,theta,c,p,ac,h)
    return y, grad
"""
end

rbf_poly = py"rbf_poly"




if Sys.islinux()
py"""
import tensorflow as tf
libPL = tf.load_op_library("./deps/PL/build/libPL.so")
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
libPL = tf.load_op_library("./deps/PL/build/libPL.dylib")
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
libPL = tf.load_op_library("./deps/PL/build/libPL.dll")
@tf.custom_gradient
def pl(x,theta,ac,h):
    y = libPL.pl(x,theta,ac,h)
    def grad(dy):
        return libPL.pl_grad(dy, y, x,theta,ac,h)
    return y, grad
"""
end

pl = py"pl"