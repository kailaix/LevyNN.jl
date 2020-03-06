function quadgauss(n::Int64, x::Float64, y::Float64, R::Float64)
    xs = zeros(n^2); ys = zeros(n^2); ws=zeros(n^2)
    if Sys.isapple()
        ccall((:quadgauss_sphere, "$(@__DIR__)/../deps/quadrature/build/libgauss.dylib"), Cvoid,
        (Cint, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}), n, R, x, y, xs, ys, ws);
    elseif Sys.islinux()
        ccall((:quadgauss_sphere, "$(@__DIR__)/../deps/quadrature/build/libgauss.so"), Cvoid,
        (Cint, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}), n, R, x, y, xs, ys, ws);
    elseif Sys.iswindows()
        ccall((:quadgauss_sphere, "$(@__DIR__)/../deps/quadrature/libgauss.ddl"), Cvoid,
        (Cint, Cdouble, Cdouble, Cdouble, Ref{Cdouble}, Ref{Cdouble}, Ref{Cdouble}), n, R, x, y, xs, ys, ws);
    end
    return [xs ys], ws
end
