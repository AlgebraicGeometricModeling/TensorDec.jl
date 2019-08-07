using LinearAlgebra
using TensorDec

v = randn(Complex{Float64},3)
#v = fill(1.0+2.0*im, 3)

X = @ring x0 x1 x2

p = (v'*X)^4

H = hankel(p,3)

w = conj(svd(H).U[:,1])

w/w[1]*v[1]-v
