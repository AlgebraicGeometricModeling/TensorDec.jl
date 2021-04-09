function [min, med, max, ts, Ns]=Numerical_exp_2_ex54_sdf(r)

n=10;
T=zeros(n,n,n);

for j= 1:n
    for j= 1:n
        for j = 1:n
            
            
            T(j,j,j)= exp(sqrt(j)+j^2*i)+(j/n)*i;
        end
    end 
end 
for j= 1:n
    for k= 1:n
        for l = 1:n
            if j == k & j~=l
                T(j,k,l)=(j/n)*i;
            end 
        end
    end 
end 
d=zeros(50,1);
t=zeros(50,1);
Ni=zeros(50,1);

for s=1:50
model = struct;
model.variables.a=complex(randn(size(T,1), r),randn(size(T,1), r));
model.variables.c=complex(randn(1, r),randn(1, r));
model.factors.A = 'a';
model.factors.C = 'c';
model.factorizations.symm.data = T;
model.factorizations.symm.cpd = {'A','A','A', 'C'};
tic,[sol, output]=sdf_nls(model,'Display', 10, 'MaxIter', 200),t(s,1)=toc;
Ni(s,1)=output.iterations;
z=sol.factors.A;
w=sol.factors.C;
T3=zeros(n,n,n);
for j= 1:r
        b=w(1,j)*outprod(z(:,j),z(:,j),z(:,j));
        T3=T3+b;
end
 d(s,1)=frob(T-T3);
end
a1=sort(d);
min=a1(1,1)
med=a1(25,1)
max=a1(50,1)
ts=sum(t)/length(t)
Ns=sum(Ni)/length(Ni)
end
