% Model parameters

% Model parameters are the following
N1 = 20; % No. of neurons in input layer 1
N2 = 20; % No. of neurons in input layer 2

% The priors on s1 and s2 are assumed to be Gaussian with zero mean and
% variance as follows
alpha_p1 = 1; % 1/variance for prior 1 (s1)
alpha_p2 = 1; % 1/variance for prior 2 (s2)

% s1 and s2 are encoded independently by neural population
% activities r1 and r2.


var_w = 1; % variance of tuning curves for input neurons

S0 = -5:10/19:5; % Preferred orientations for input neurons
% Same for both sets of populations
