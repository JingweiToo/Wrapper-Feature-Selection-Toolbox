%[2014]-"Grey wolf optimizer"

% (9/12/2020)

function GWO = jGreyWolfOptimizer(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
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
fit = zeros(1,N);
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > thres),opts);
end
% Sort fitness  
[~, idx] = sort(fit,'ascend');  
% Update alpha, beta & delta 
Xalpha = X(idx(1),:);
Xbeta  = X(idx(2),:);
Xdelta = X(idx(3),:);
Falpha = fit(idx(1));
Fbeta  = fit(idx(2)); 
Fdelta = fit(idx(3));
% Pre
curve = zeros(1,max_Iter);
curve(1) = Falpha; 
t = 2; 
% Iterations
while t <= max_Iter
	% Coefficient decreases linearly from 2 to 0 
  a = 2 - t * (2 / max_Iter); 
  for i = 1:N
    for d = 1:dim
      % Parameter C (3.4)
      C1 = 2 * rand();
      C2 = 2 * rand();
      C3 = 2 * rand();
      % Compute Dalpha, Dbeta & Ddelta (3.5)
      Dalpha = abs(C1 * Xalpha(d) - X(i,d)); 
      Dbeta  = abs(C2 * Xbeta(d) - X(i,d));
      Ddelta = abs(C3 * Xdelta(d) - X(i,d));
      % Parameter A (3.3)
      A1 = 2 * a * rand() - a; 
      A2 = 2 * a * rand() - a; 
      A3 = 2 * a * rand() - a;
      % Compute X1, X2 & X3 (3.6) 
      X1 = Xalpha(d) - A1 * Dalpha;
      X2 = Xbeta(d) - A2 * Dbeta; 
      X3 = Xdelta(d) - A3 * Ddelta;
      % Update wolf (3.7)
      X(i,d) = (X1 + X2 + X3) / 3; 
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    % Fitness 
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Update alpha, beta & delta
    if fit(i) < Falpha
      Falpha = fit(i);
      Xalpha = X(i,:);
    end
    if fit(i) < Fbeta && fit(i) > Falpha
      Fbeta = fit(i);
      Xbeta = X(i,:);
    end
    if fit(i) < Fdelta && fit(i) > Falpha && fit(i) > Fbeta
      Fdelta = fit(i); 
      Xdelta = X(i,:);
    end
  end
  curve(t) = Falpha;
  fprintf('\nIteration %d Best (GWO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim; 
Sf    = Pos((Xalpha > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
GWO.sf = Sf; 
GWO.ff = sFeat; 
GWO.nf = length(Sf); 
GWO.c  = curve;
GWO.f  = feat;
GWO.l  = label;
end



