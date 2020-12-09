%[2016]-"The whale optimization algorithm"

% (9/12/2020)

function WOA = jWhaleOptimizationAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
b     = 1;     % constant

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'b'), b = opts.b; end 
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
  % Global best
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG; 
t = 2; 
while t <= max_Iter
	% Define a, linearly decreases from 2 to 0 
  a = 2 - t * (2 / max_Iter);
  for i = 1:N
    % Parameter A (2.3)
    A = 2 * a * rand() - a;
    % Paramater C (2.4)
    C = 2 * rand();
    % Parameter p, random number in [0,1]
    p = rand();
    % Parameter l, random number in [-1,1]
    l = -1 + 2 * rand();  
    % Whale position update (2.6)
    if p  < 0.5
      % {1} Encircling prey
      if abs(A) < 1
        for d = 1:dim
          % Compute D (2.1)
          Dx     = abs(C * Xgb(d) - X(i,d));
          % Position update (2.2)
          X(i,d) = Xgb(d) - A * Dx;
        end
      % {2} Search for prey
      elseif abs(A) >= 1
        for d = 1:dim
          % Select a random whale
          k      = randi([1,N]);
          % Compute D (2.7)
          Dx     = abs(C * X(k,d) - X(i,d));
          % Position update (2.8)
          X(i,d) = X(k,d) - A * Dx;
        end
      end
    % {3} Bubble-net attacking 
    elseif p >= 0.5
      for d = 1:dim
        % Distance of whale to prey
        dist   = abs(Xgb(d) - X(i,d));
        % Position update (2.5)
        X(i,d) = dist * exp(b * l) * cos(2 * pi * l) + Xgb(d);
      end
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    X(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    % Fitness 
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Global best
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (WOA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
WOA.sf = Sf; 
WOA.ff = sFeat;
WOA.nf = length(Sf);
WOA.c  = curve;
WOA.f  = feat;
WOA.l  = label;
end





