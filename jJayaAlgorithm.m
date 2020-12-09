%[2016]-"Jaya: A simple and new optimization algorithm for solving 
%constrained and unconstrained optimization problems"

% (9/12/2020)

function JA = jJayaAlgorithm(feat,label,opts)
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
% Initial (26)
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
  % Best
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Pre
Xnew = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Iteration
while t <= max_Iter
  % Identify best & worst in population
  [~, idxB] = min(fit); 
  Xbest     = X(idxB,:);
  [~, idxW] = max(fit);
  Xworst    = X(idxW,:);
  % Start
  for i = 1:N
    for d = 1:dim
      % Random numbers
      r1 = rand();
      r2 = rand();
      % Update (1)
      Xnew(i,d) = X(i,d) + r1 * (Xbest(d) - abs(X(i,d))) - ...
        r2 * (Xworst(d) - abs(X(i,d)));
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end 
  % Fitness
  for i = 1:N
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Greedy selection 
    if Fnew < fit(i)
      fit(i) = Fnew;
      X(i,:) = Xnew(i,:);
    end
    % Best
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (JA)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
JA.sf = Sf;
JA.ff = sFeat;
JA.nf = length(Sf); 
JA.c  = curve; 
JA.f  = feat;
JA.l  = label;
end



