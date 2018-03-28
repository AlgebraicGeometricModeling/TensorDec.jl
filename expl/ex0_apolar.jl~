using TensorDec


X = @ring x0 x1 x2 
n = length(X)
d = 4
r = 4

println("Symmetric tensor: dim ", n, "  degree ",d, "  rank ",r)
Xi = rand(r,n)
w = fill(1.0,r)
T = tensor(w,Xi,X, d)

println("Hilbert fct: ", hilbert(T))
k = 2
H = hankel(T,k)

P = perp(T,k)
Pa = map(t->subs(t,x0=>1.), P)
println("Perp ", k, " : ", size(H),"  kernel ",length(P))

#Xis = solve_macaulay(P,X,3)
#ws = weights(T,Xis)

#T0 = tensor(ws,Xis,X,d)
#norm(T-T0,Inf)
