using TensorDec, DynamicPolynomials
X = @polyvar x0 x1 x2
d=4; F = (x0+x1+0.75x2)^d + 1.5*(x0-x1)^d -2.0*(x0-x2)^d
w, Xi = decompose(F)

println("w = ",w)
println("Xi = ",Xi)

T0 = tensor(w, Xi, X, d)

er = norm_apolar(F-T0)

println("Error: $er")
