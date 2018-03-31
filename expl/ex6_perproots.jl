using AlgebraicSolvers
using TensorDec

X = @ring x0 x1 x2 
n = length(X)
d = 4
r = 4

println("Symmetric tensor: dim ", n, "  degree ",d, "  rank ",r)
Xi = rand(n,r)
w = fill(1.0,r)
F = tensor(w,Xi,X, d)

println("Hilbert fct: ", hilbert(F))
k = 2
H = hankel(F,k)

P = perp(F,k)
Pa = map(t->subs(t,x0=>1.), P)
println("Perp ", k, " : ", size(H),"  kernel ",length(P))

Xis = solve_macaulay(P,X,5)

ws = weights(F,Xis')
F1 = tensor(ws,Xis', X,d)
norm(F-F1,Inf)
