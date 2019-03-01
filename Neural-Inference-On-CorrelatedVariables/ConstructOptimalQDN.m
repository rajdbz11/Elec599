function r3 = ConstructOptimalQDN(r1,r2,params)

a1 = params.a1;
a2 = params.a2;
b1 = params.b1;
b2 = params.b2;
alpha_p1 = params.alpha_p1;
alpha_p2 = params.alpha_p2;
a3_d = params.a3_d;
b3_d = params.b3_d;
c3_d = params.c3_d;
f3   = params.f3;

r1 = r1';
r2 = r2';

Den = a1'*r1 + a2'*r2 + alpha_p1 + alpha_p2;
% Den = 1;
A   = (a1'*r1 + alpha_p1)*(a2'*r2 + alpha_p2)/Den;
B   = ((b1'*r1)*(a2'*r2 + alpha_p2) + (b2'*r2)*(a1'*r1 + alpha_p1))/Den;
r3  = A*a3_d + B*b3_d + f3*c3_d;
