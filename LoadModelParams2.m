% Model parameters

NN = 20;
var_w = 1; % variance of tuning curves for input neurons

S0 = linspace(-5,5,NN); % Preferred orientations for input neurons

% Parameters for stimulus dependent kernel as specified in Beck et al.
var_3w = 1;
theta1 = 5; % previously 1/20
theta2 = 10;
f3     = 1; % Arbitrary scalar

% Requirements for optimal model construction
ii  = (1:NN)'; % vector index
i0 = (NN+1)/2;

a3_bias = mean(exp(-2*((ii-i0).^2)/((NN^2)*var_3w)));
a3      = theta1*exp(-2*((ii-i0).^2)/((NN^2)*var_3w)) - theta1*a3_bias;
b3      = theta1*(ii-i0).*exp(-2*((ii-i0).^2)/((NN^2)*var_3w))/NN;

% Computing a3_d and b3_d
Za = a3'*a3;
Zb = b3'*b3;
a3_d = a3/Za;
b3_d = b3/Zb;
c3_d = ones(NN,1)/theta2;

a1 = ones(NN,1)/var_w;
% a2 = ones(NN,1)/var_w;
b1 = S0'/var_w;
% b2 = S0'/var_w;

clear NN var_3w theta1 theta2 ii i0 a3_bias Za Zb

% optimal QDN params
QDNParams = [];
QDNParams.a = a1;
% QDNParams.a2 = a2;
QDNParams.b = b1;
% QDNParams.b2 = b2;
QDNParams.at = a3;
QDNParams.bt = b3;
QDNParams.at_d = a3_d;
QDNParams.bt_d = b3_d;
QDNParams.ct_d = c3_d;
QDNParams.ft   = f3;

clear a1 a2 a3 a3_d b1 b2 b3 b3_d c3_d f3;