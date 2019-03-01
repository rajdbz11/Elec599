clear; close all
% Define the parameters of the bivariate Gaussian distribution
% Parameters of the tuning curves
Var1    = 5;
Mu1     = 0;  
Var2    = 5;
Mu2     = 0;
Var3    = 5;
Mu3     = 0;
Ro13    = 0; %-0.15;
Ro23    = 0; %0.75;

load QDNParams;

% Preferred orientations
N = 20;
PO                  = linspace(-5,5,N)';
[POMatA,POMatB]     = meshgrid(PO,PO); % We have a 2D grid of preferred orientations now
POVecA              = POMatA(:);
POVecB              = POMatB(:);

% Present some stimulus value now
s1Val   = 0;
s2Val   = 0;
s3Val   = 0;

% -----------  First do it for the pairwise factors  ----------------------
CovMat13    = [Var1 Ro13*sqrt(Var1*Var3); Ro13*sqrt(Var1*Var3) Var3];
CovMat23    = [Var2 Ro23*sqrt(Var2*Var3); Ro23*sqrt(Var2*Var3) Var3];
gainval     = 1;
K13         = 2*pi*sqrt(Var1*Var3*(1-Ro13^2))*gainval;
K23         = 2*pi*sqrt(Var2*Var3*(1-Ro23^2))*gainval;

% Generate the mean firing rates
F13     = K13*mvnpdf(repmat([s1Val, s3Val],length(POVecA),1), [POVecA, POVecB], CovMat13);
F23     = K23*mvnpdf(repmat([s2Val, s3Val],length(POVecA),1), [POVecA, POVecB], CovMat23);

% Generate one instance of neural activity
R13     = poissrnd(F13);
R23     = poissrnd(F23);

% Now to get the posterior distribution
% First define all the "kernel" vectors
a11 = ones(N^2,1)/(1-Ro13^2)/Var1;
a33 = ones(N^2,1)/(1-Ro13^2)/Var3;
a13 = Ro13*ones(N^2,1)/(1-Ro13^2)/sqrt(Var1*Var3);
a1  = POVecA/(1-Ro13^2)/Var1 - Ro13*POVecB/(1-Ro13^2)/sqrt(Var1*Var3);
a3  = POVecB/(1-Ro13^2)/Var3 - Ro13*POVecA/(1-Ro13^2)/sqrt(Var1*Var3);

b22 = ones(N^2,1)/(1-Ro23^2)/Var2;
b33 = ones(N^2,1)/(1-Ro23^2)/Var3;
b23 = Ro23*ones(N^2,1)/(1-Ro23^2)/sqrt(Var2*Var3);
b2  = POVecA/(1-Ro23^2)/Var2 - Ro23*POVecB/(1-Ro23^2)/sqrt(Var2*Var3);
b3  = POVecB/(1-Ro23^2)/Var3 - Ro23*POVecA/(1-Ro23^2)/sqrt(Var2*Var3);

A11 = a11'*R13;
A33 = a33'*R13;
A13 = a13'*R13;
A1  = a1'*R13;
A3  = a3'*R13;

B22 = b22'*R23;
B33 = b33'*R23;
B23 = b23'*R23;
B2  = b2'*R23;
B3  = b3'*R23;

% Stim range for plotting the posterior distributions
s1Vec = -5:0.01:5;
s2Vec = -5:0.01:5;
[SMatA, SMatB] = meshgrid(s1Vec,s2Vec);
SVecA = SMatA(:);
SVecB = SMatB(:);

% Computing the pairwise factor terms
Phi13Vec = exp(-0.5*A11*SVecA.^2 + A1*SVecA + A13*SVecA.*SVecB -0.5*A33*SVecB.^2 + A3*SVecB);
Phi13Mat = reshape(Phi13Vec,length(s1Vec),length(s1Vec));

Phi23Vec = exp(-0.5*B22*SVecA.^2 + B2*SVecA + B23*SVecA.*SVecB -0.5*B33*SVecB.^2 + B3*SVecB);
Phi23Mat = reshape(Phi23Vec,length(s1Vec),length(s1Vec));

% % Plotting the posteriors for visualization

% figure; mesh(SMatA, SMatB, Phi13Mat);
% figure; contour(SMatA,SMatB, log(Phi13Mat)); hold on
% plot(s1Val, s3Val, 'rx', 'MarkerSize',10,'LineWidth',2);
% grid on
% 
% figure; contour(SMatA,SMatB, log(Phi23Mat)); hold on
% plot(s2Val, s3Val, 'rx', 'MarkerSize',10,'LineWidth',2);
% grid on

% --------------------- Now get the singleton factors ---------------------
s1Val   = 2;
s2Val   = -1;
s3Val   = -2;


G = [1; 1; 1]; % Gain values for each population
% Parameters of the tuning curves
var_w1 = 5;
var_w2 = 5;
var_w3 = 5;

f1 = G(1)*exp(-((s1Val - PO).^2)/(2*var_w1));
f2 = G(2)*exp(-((s2Val - PO).^2)/(2*var_w2));
f3 = G(3)*exp(-((s3Val - PO).^2)/(2*var_w3));

% Now to generate one instance of input populations
r1 = poissrnd(f1); 
r2 = poissrnd(f2); 
r3 = poissrnd(f3); 

c1 = ones(N,1)/var_w1;
c2 = ones(N,1)/var_w2;
c3 = ones(N,1)/var_w3;
d1 = PO/var_w1;
d2 = PO/var_w2;
d3 = PO/var_w3;

C1 = c1'*r1; C2 = c2'*r2; C3 = c3'*r3;
D1 = d1'*r1; D2 = d2'*r2; D3 = d3'*r3;

% Singleton factors
ph1Vec = exp(-0.5*C1*s1Vec.^2 + D1*s1Vec);
ph2Vec = exp(-0.5*C2*s1Vec.^2 + D2*s1Vec);
ph3Vec = exp(-0.5*C3*s1Vec.^2 + D3*s1Vec);

% figure; 
% plot(s1Vec, ph1Vec/sum(ph1Vec), 'b.-'); hold on
% plot(s1Vec, ph2Vec/sum(ph2Vec), 'r.-');
% plot(s1Vec, ph3Vec/sum(ph3Vec), 'c.-');

% Now we have all the factors defined

% ------------ Let us get to the messages now -----------------------------
X13 = - A13^2/(C1 + A11);
Y13 = A13*(A1 + D1)/(C1 + A11);

X23 = - B23^2/(C2 + B22);
Y23 = B23*(B2 + D2)/(C2 + B22);

% Get the posteriors from these messages
m13Vec = exp(-0.5*(A33 + X13)*s1Vec.^2 + (A3 + Y13)*s1Vec);
m23Vec = exp(-0.5*(B33 + X23)*s1Vec.^2 + (B3 + Y23)*s1Vec);


% The posterior for the marginal population
AFinal = C3 + A33 + X13 + B33 + X23;
BFinal = D3 + A3 + Y13 + B3 + Y23;

PFinalVec = exp(-0.5*AFinal*s1Vec.^2 + BFinal*s1Vec);

% % Plotting the posteriors
% figure;
% plot(s1Vec, m13Vec/sum(m13Vec), 'b','LineWidth',2.5); hold on
% plot(s1Vec, m23Vec/sum(m23Vec), 'r','LineWidth',2.5);
% plot(s1Vec, ph3Vec/sum(ph3Vec), 'c','LineWidth',2.5);
% plot(s1Vec, PFinalVec/sum(PFinalVec), 'k','LineWidth',2.5);

% Now to generate the populations
DN   = 100;
M13  = (A33 + X13)*QDNparams.a3_d/DN + (A3 + Y13)*QDNparams.b3_d/DN + QDNparams.f3*QDNparams.c3_d;
M23  = (B33 + X23)*QDNparams.a3_d/DN + (B3 + Y23)*QDNparams.b3_d/DN + QDNparams.f3*QDNparams.c3_d;

RFinal  = (AFinal)*QDNparams.a3_d/DN + (BFinal)*QDNparams.b3_d/DN + QDNparams.f3*QDNparams.c3_d;

% Now make the Poisson draws
M13_P = poissrnd(M13 - min(M13));
M23_P = poissrnd(M23 - min(M23));

RFinal_P = poissrnd(RFinal - min(RFinal));

% Get the posterior distributions from these Poisson draws now
m13Vec_P = exp(-0.5*(M13_P'*QDNparams.a3_d/DN/norm(QDNparams.a3_d/DN)^2)*s1Vec.^2 + (M13_P'*QDNparams.b3_d/DN/norm(QDNparams.b3_d/DN)^2)*s1Vec);
m23Vec_P = exp(-0.5*(M23_P'*QDNparams.a3_d/DN/norm(QDNparams.a3_d/DN)^2)*s1Vec.^2 + (M23_P'*QDNparams.b3_d/DN/norm(QDNparams.b3_d/DN)^2)*s1Vec);

PFinalVec_P = exp(-0.5*(RFinal_P'*QDNparams.a3_d/DN/norm(QDNparams.a3_d/DN)^2)*s1Vec.^2 + (RFinal_P'*QDNparams.b3_d/DN/norm(QDNparams.b3_d/DN)^2)*s1Vec);

% % Plotting these newly obtained posteriors 
% figure;
% plot(s1Vec, m13Vec_P/sum(m13Vec_P), 'b','LineWidth',2.5); hold on
% plot(s1Vec, m23Vec_P/sum(m23Vec_P), 'r','LineWidth',2.5);
% plot(s1Vec, ph3Vec/sum(ph3Vec), 'c','LineWidth',2.5);
% plot(s1Vec, PFinalVec_P/sum(PFinalVec_P), 'k','LineWidth',2.5);


figure;
h = area(s1Vec, m13Vec_P/sum(m13Vec_P), 'FaceColor','b'); hold on
child = get(h,'Children');
set(child,'FaceAlpha',0.5)
h = area(s1Vec, m23Vec_P/sum(m23Vec_P), 'FaceColor','r');
child = get(h,'Children');
set(child,'FaceAlpha',0.5)
h = area(s1Vec, ph3Vec/sum(ph3Vec), 'FaceColor','g');
child = get(h,'Children');
set(child,'FaceAlpha',0.5)
h = area(s1Vec, PFinalVec_P/sum(PFinalVec_P), 'FaceColor','k');
child = get(h,'Children');
set(child,'FaceAlpha',0.25)
