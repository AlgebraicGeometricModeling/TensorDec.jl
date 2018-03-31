using TensorDec

r  = 3
w0 = rand(r)
A0 = rand(3,r)
B0 = rand(5,r)
C0 = rand(4,r)

T0 = tensor(w0, A0, B0, C0)

w, A, B, C = decompose(T0, eps_rkf(1.e-10), mode=2)

T = tensor(w, A, B, C)

norm(T-T0)
