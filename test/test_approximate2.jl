using LinearAlgebra, DynamicPolynomials, TensorDec

n = 4
X = (@polyvar x[1:n])[1]

d = 3

r0 = 10
w  = randn(r0)
Xi = randn(n,r0)

T0 = tensor(w, Xi, X, d)

r = 4

Nt = 5

w0, V0 = decompose(T0,cst_rkf(r))

print("RGN :")
w1r, V1r, Info = approximate(T0, w0, V0; mthd = :RGN)

T1r = tensor(w1r,V1r,X,d)
print(" ",norm_apolar(T0-T1r))
println()


print("RNE :")
w2r, V2r, Info = approximate(T0, w0, V0; mthd = :RNE)
T2r = tensor(w2r,V2r,X,d)
print(" ",norm_apolar(T0-T2r))
println()

w3, V3, I = approximate(T0, w0, V0; mthd = :SPM)

T3 = tensor(w3,V3,X,d)
println("SPM  : ",norm_apolar(T0-T3))


