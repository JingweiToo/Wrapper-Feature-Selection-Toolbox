%% Wrapper Feature Selection Demostration 

% Source code of the FS methods are written based on pseudocode 
% There are more than 40 wrapper FS methods are offered 
% You may open < List_Method.m file > to check all available methods

%---Usage-------------------------------------------------------------
% If you wish to use 'PSO' (see example 1) then you write
% FS = jfs('pso',feat,label,opts);

% If you want to use 'SMA' (see example 2) then you write
% FS = jfs('sma',feat,label,opts);

% * All methods have different name/abbrevation (refer jfs.m file)


%---Input-------------------------------------------------------------
% feat   : Feature vector matrix (Instances x Features)
% label  : Label matrix (Instances x 1)
% opts   : Parameter settings 
% opts.N : Number of solutions / population size (* for all methods)
% opts.T : Maximum number of iterations (* for all methods)
% opts.k : Number of k in k-nearest neighbor 

% Some methods have their specific parameters (example: PSO, GA, DE) 
% if you do not set them then they will define as default settings
% * you may open the < m.file > to view or change the parameters
% * you may use 'opts' to set the parameters of method (see example 1)
% * you may also change the < jFitnessFunction.m file >


%---Output------------------------------------------------------------
% FS    : Feature selection model (It contains several results)
% FS.sf : Index of selected features
% FS.ff : Selected features
% FS.nf : Number of selected features
% FS.c  : Convergence curve
% Acc   : Accuracy of validation model


%% Example 1: Particle Swarm Optimization (PSO) 
clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Parameters of PSO
opts.c1 = 2;
opts.c2 = 2;
opts.w  = 0.9;
% Load dataset
load ionosphere.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('pso',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% Accuracy  
Acc    = jknn(feat(:,sf_idx),label,opts); 
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('PSO');


%% Example 2: Slime Mould Algorithm (SMA) 
clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N  = 10;     % number of solutions
opts.T  = 100;    % maximum number of iterations
% Load dataset
load ionosphere.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('sma',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% Accuracy  
Acc    = jknn(feat(:,sf_idx),label,opts); 
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('SMA');


%% Example 3: Whale Optimization Algorithm (WOA) 
clear, clc, close;
% Number of k in K-nearest neighbor
opts.k = 5; 
% Ratio of validation data
ho = 0.2;
% Common parameter settings 
opts.N = 10;     % number of solutions
opts.T = 100;    % maximum number of iterations
% Parameter of WOA
opts.b = 1;
% Load dataset
load ionosphere.mat; 
% Divide data into training and validation sets
HO = cvpartition(label,'HoldOut',ho); 
opts.Model = HO; 
% Perform feature selection 
FS     = jfs('woa',feat,label,opts);
% Define index of selected features
sf_idx = FS.sf;
% Accuracy  
Acc    = jknn(feat(:,sf_idx),label,opts); 
% Plot convergence
plot(FS.c); grid on;
xlabel('Number of Iterations'); 
ylabel('Fitness Value'); 
title('WOA');




