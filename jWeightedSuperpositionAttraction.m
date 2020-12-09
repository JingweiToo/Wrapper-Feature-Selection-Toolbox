%[2017]-"Weighted Superposition Attraction (WSA): A swarm 
%intelligence algorithm for optimization problems – Part 1: 
%Unconstrained optimization"

% (8/12/2020)

function WSA = jWeightedSuperpositionAttraction(feat,label,opts)
% Parameters
lb     = 0;
ub     = 1; 
thres  = 0.5;
tau    = 0.8;      % constant 
sl     = 0.035;    % step length
phi    = 0.001;    % constant
lambda = 0.75;     % constant

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'tau'), tau = opts.tau; end
if isfield(opts,'sl'), sl = opts.sl; end
if isfield(opts,'phi'), phi = opts.phi; end
if isfield(opts,'lambda'), lambda = opts.lambda; end
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
% Fitness
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
  % Best update
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2;
% Iterations
while t <= max_Iter
  % Rank solution based on fitness
  [fit, idx] = sort(fit,'ascend'); 
  X          = X(idx,:); 
  % {1} Target point determination: Figure 2
  w    = zeros(1,N);
  Xtar = zeros(1,dim);
  for i = 1:N
    % Assign weight based on rank
    w(i) = i ^ (-1 * tau);
    % Create target 
    for d = 1:dim
      Xtar(d) = Xtar(d) + X(i,d) * w(i);
    end
  end
  % Boundary
  Xtar(Xtar > ub) = ub; 
  Xtar(Xtar < lb) = lb;
  % Fitness
  fitT = fun(feat,label,(Xtar > thres),opts);
  % Best update
  if fitT < fitG
    fitG = fitT;
    Xgb  = Xtar;
  end
  % {2} Compute search direction: Figure 4 
  gap    = zeros(N,dim); 
  direct = zeros(N,dim);
  for i = 1:N
    if fit(i) >= fitT
      for d = 1:dim
        % Compute gap
        gap(i,d)    = Xtar(d) - X(i,d);
        % Compute direction
        direct(i,d) = sign(gap(i,d));
      end
    elseif fit(i) < fitT
      if rand() < exp(fit(i) - fitT)
        for d = 1:dim
          % Compute gap
          gap(i,d)    = Xtar(d) - X(i,d);
          % Compute direction
          direct(i,d) = sign(gap(i,d));
        end
      else
        for d = 1:dim
          % Compute direction
          direct(i,d) = sign(-1 + (1 + 1) * rand());
        end
      end
    end
  end
  % Compute step sizing function (2)
  if rand() <= lambda
    sl = sl - exp(t / (t - 1)) * phi * sl;
  else
    sl = sl + exp(t / (t - 1)) * phi * sl;
  end
  % {3} Neighbor generation: Figure 7
  for i = 1:N
    for d = 1:dim
      % Update (1)
      X(i,d) = X(i,d) + sl * direct(i,d) * abs(X(i,d));
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    X(i,:) = XB;
  end 
  % Fitness
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best update
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (WSA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
WSA.sf = Sf;
WSA.ff = sFeat; 
WSA.nf = length(Sf);
WSA.c  = curve; 
WSA.f  = feat;
WSA.l  = label;
end


