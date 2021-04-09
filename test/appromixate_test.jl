using LinearAlgebra, MultivariateSeries, DynamicPolynomials, TensorDec

X = @polyvar x y z
n = length(X)

d = 3

r0 = 6
w  = randn(r0)
Xi = randn(n,r0)

T0 = tensor(w, Xi, X, d)

r = 3

w1, V1, I = approximate(T0, r; mthd = :RGN)

T1 = tensor(w1,V1,X,d)
println("T1: ",norm(T0-T1))

w2, V2, I = approximate(T0, r; mthd = :RNE)

T2 = tensor(w2,V2,X,d)
println("T2: ",norm(T0-T2))
