using LinearAlgebra, MultivariateSeries, DynamicPolynomials, TensorDec

n = 4
X = (@polyvar x[1:n])[1]

d = 3

r0 = 10
w  = randn(r0)
Xi = randn(n,r0)

T0 = tensor(w, Xi, X, d)

r = 3

Nt = 5

print("RGNr :")
for i in 1:Nt
    w1r, V1r, Info = approximate(T0, r; mthd = :RGN)
    T1r = tensor(w1r,V1r,X,d)
    print(" ",norm_apolar(T0-T1r))
end
println()

w1, V1, I = approximate(T0, r; mthd = :RGN, sdm = :NRnd)
T1 = tensor(w1,V1,X,d)
println("RGNn : ",norm_apolar(T0-T1))


print("RGNr :")
for i in 1:Nt
    w2r, V2r, Info = approximate(T0, r; mthd = :RNE)
    T2r = tensor(w2r,V2r,X,d)
    print(" ",norm_apolar(T0-T2r))
end
println()

w2, V2, I = approximate(T0, r; mthd = :RNE, sdm = :NRnd)
T2 = tensor(w2,V2,X,d)
println("RNEn : ",norm_apolar(T0-T2))

w3, V3, I = approximate(T0, r; mthd = :SPM)

T3 = tensor(w3,V3,X,d)
println("SPM  : ",norm_apolar(T0-T3))

print("RNERr:")
for i in 1:Nt
    try 
        w4r, V4r, Info = approximate(T0, r; mthd = :RNER)
        T4r = tensor(w4r,V4r,X,d)
        print(" ",norm_apolar(T0-T4r))
    catch
        print(" ******")
    end
end
println()

w4, V4, I = approximate(T0, r; mthd = :RNER, sdm = :NRnd)
T4 = tensor(w4,V4,X,d)
println("RNERn: ",norm_apolar(T0-T4))
