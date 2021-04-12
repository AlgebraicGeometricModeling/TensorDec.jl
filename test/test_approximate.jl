using LinearAlgebra, MultivariateSeries, DynamicPolynomials, TensorDec

n = 4
X = (@polyvar x[1:n])[1]

d = 3

r0 = 10
w  = randn(r0)
Xi = randn(n,r0)

T0 = tensor(w, Xi, X, d)

r = 3


w1r, V1r, I = approximate(T0, r; mthd = :RGN)
T1r = tensor(w1r,V1r,X,d)
println("RGNr: ",norm_apolar(T0-T1r))


w1, V1, I = approximate(T0, r; mthd = :RGN, sdm = :NRnd)
T1 = tensor(w1,V1,X,d)
println("RGNn: ",norm_apolar(T0-T1))

w2r, V2r, I = approximate(T0, r; mthd = :RNE)
T2r = tensor(w2r,V2r,X,d)
println("RNEr: ",norm_apolar(T0-T2r))

w2, V2, I = approximate(T0, r; mthd = :RNE, sdm = :NRnd)
T2 = tensor(w2,V2,X,d)
println("RNEn: ",norm_apolar(T0-T2))



w3, V3, I = approximate(T0, r; mthd = :SPM)

T3 = tensor(w3,V3,X,d)
println("SPM:  ",norm_apolar(T0-T3))
