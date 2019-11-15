using LinearAlgebra
using TensorDec

X = @ring x0 x1 x2

n = length(X)
d = 4
r = 3

println("Symmetric tensor: dim ", n, "  degree ",d, "  rank ",r)
Xi0 = rand(n,r)
w0 = fill(1.0,r)
T = tensor(w0,Xi0,X, d)


w, Xi = decompose(T)

println("w=",w)
println("Xi=",Xi)

T1 = tensor(w,Xi,X,d);
println("Error: ", norm(T-T1,Inf))
