%[2018]-"Emperor penguin optimizer: A bio-inspired algorithm for
%engineering problems"

% (8/12/2020)

function EPO = jEmperorPenguinOptimizer(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
M     = 2;     % movement parameter
f     = 3;     % control parameter
l     = 2;     % control parameter

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'M'), M = opts.M; end 
if isfield(opts,'f'), f = opts.f; end 
if isfield(opts,'l'), l = opts.l; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction;
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
	end
end
% Pre
fit  = zeros(1,N); 
fitG = inf;

curve = inf; 
t = 1;
% Iterations
while t <= max_Iter
  for i = 1:N
    % Fitness 
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best solution
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  % Generate radius in [0,1]
  R = rand();
  % Time (7)
  if R > 1
    T0 = 0;
  else
    T0 = 1;
  end
  % Temperature profile (7)
  T = T0 - (max_Iter / (t - max_Iter));
  for i = 1:N
    for d = 1:dim
      % Pgrid (10)
      P_grid = abs(Xgb(d) - X(i,d));
      % Vector A (9)
      A = (M * (T + P_grid) * rand()) - T;
      % Vector C (11)
      C = rand();
      % Compute function S (12)
      S = sqrt(f * exp(t / l) - exp(-t)) ^ 2;
      % Distance (8)
      Dep = abs(S * Xgb(d) - C * X(i,d));
      % Position update (13)
      X(i,d) = Xgb(d) - A * Dep;
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (EPO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
EPO.sf = Sf; 
EPO.ff = sFeat; 
EPO.nf = length(Sf);
EPO.c  = curve;
EPO.f  = feat;
EPO.l  = label;
end



