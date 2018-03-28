using TensorDec

A0 = rand(4,2)
B0 = rand(3,2)
C0 = rand(5,2)
w0 = rand(2)

T0 = tensor(w0, A0, B0, C0)

w, A, B, C = decompose(T0)

T = tensor(w,A,B,C)

norm(reshape(T-T0,length(T0)))
