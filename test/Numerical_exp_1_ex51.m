function [Ms,ts,Ns]=Numerical_exp_1_ex51(n)
R=1;
T=zeros(n,n,n);

for j= 1:n
    for k= 1:n
        for l = 1:n
            
            
     T(j,k,l)=((-1)^j)/(j)+((-1)^k)/(k)+((-1)^l)/(l);
            
    end
end 
end 
M=zeros(50,1);
t=zeros(50,1);
Ni=zeros(50,1);
z=zeros(n,50);
z1=zeros(50,1);
for s=1:50
model = struct;
model.variables.a=randn(size(T,1), R);
model.variables.c=randn(1, R);
model.factors.A = 'a';
model.factors.C = 'c';
model.factorizations.symm.data = T;
model.factorizations.symm.cpd = {'A','A','A', 'C'};
tic,[sol, output]=ccpd_nls(model,'Display', 10, 'MaxIter', 200),t(s,1)=toc;
Ni(s,1)=output.iterations;
z(:,s)=sol{1};
z1(s,1)=sol{2}*(norm(z(:,s)))^3;
M(s,1)=abs(z1(s,1));
end 
Ms=sum(M)/length(M)
ts=sum(t)/length(t)
Ns=sum(Ni)/length(Ni)
end