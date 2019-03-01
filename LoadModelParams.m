% Model parameters

% Model parameters are the following
N1 = 20; % No. of neurons in input layer 1
N2 = 20; % No. of neurons in input layer 2

% The priors on s1 and s2 are assumed to be Gaussian with zero mean and
% variance as follows
alpha_p1 = 1; % 1/variance for prior 1 (s1)
alpha_p2 = 1; % 1/variance for prior 2 (s2)
% Variance of prior for s3; assuming independence/uncorrelated s1, s2
var_s3 = (1/alpha_p1) + (1/alpha_p2);


% The prior for s3 is assumed to be a Gaussain with zero mean and variance
% the sum of variances of s1 and s2. This is assuming independence of s1
% and s2, which is necessary. One thing is for certain with this model,
% is that s1 and s2 are encoded independently by neural population
% activities r1 and r2.


var_w = 1; % variance of tuning curves for input neurons

S0 = -5:10/19:5; % Preferred orientations for input neurons

% Parameters for stimulus dependent kernel as specified in Beck et al.
var_3w = 1;
theta1 = 1/20;
theta2 = 10;
f3     = 1; % Arbitrary scalar

% Requirements for optimal model construction
i  = (1:N1)'; % vector index
i0 = (N1+1)/2;

a3_bias = mean(exp(-2*((i-i0).^2)/((N1^2)*var_3w)));
a3      = theta1*exp(-2*((i-i0).^2)/((N1^2)*var_3w)) - theta1*a3_bias;
b3      = theta1*(i-i0).*exp(-2*((i-i0).^2)/((N1^2)*var_3w))/N1;

% Computing a3_d and b3_d
Za = a3'*a3;
Zb = b3'*b3;
a3_d = a3/Za;
b3_d = b3/Zb;
c3_d = ones(N1,1)/theta2;

a1 = ones(N1,1)/var_w;
a2 = ones(N1,1)/var_w;
b1 = S0'/var_w;
b2 = S0'/var_w;

% optimal QDN params
QDNparams = [];
QDNparams.a1 = a1;
QDNparams.a2 = a2;
QDNparams.b1 = b1;
QDNparams.b2 = b2;
QDNparams.alpha_p1 = alpha_p1;
QDNparams.alpha_p2 = alpha_p2;
QDNparams.a3_d = a3_d;
QDNparams.b3_d = b3_d;
QDNparams.c3_d = c3_d;
QDNparams.f3   = f3;