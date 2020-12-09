%[2012]-"A new fruit fly optimization algorithm: Taking the financial 
%distress model as an example"  

% (9/12/2020)

function FOA = jFruitFlyOptimizationAlgorithm(feat,label,opts)
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
X = zeros(N,dim);
Y = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
    Y(i,d) = lb + (ub - lb) * rand();
  end
end  
% Compute solution
S = zeros(N,dim);
for i = 1:N
  for d = 1:dim
    % Distance between X and Y axis
    dist = sqrt(X(i,d) ^ 2 + Y(i,d) ^ 2);
    % Solution
    S(i,d) = 1 / dist;
  end
  % Boundary
  SB = S(i,:); SB(SB > ub) = ub; SB(SB < lb) = lb; 
  S(i,:) = SB;
end
% Pre
fit   = zeros(1,N);
fitG  = inf;
curve = inf;
t = 1; 
% Iterations
while t <= max_Iter 
  % Fitness
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,(S(i,:) > thres),opts);
    % Update better solution
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = S(i,:);
      % Update X & Y
      Xb   = X(i,:);
      Yb   = Y(i,:);
    end
  end
	for i = 1:N
    for d = 1:dim
      % Random in [-1,1]
      r1 = -1 + 2 * rand(); 
      r2 = -1 + 2 * rand();
      % Compute new X & Y
      X(i,d) = Xb(d) + (ub - lb) * r1;
      Y(i,d) = Yb(d) + (ub - lb) * r2;
      % Distance between X and Y axis
      dist   = sqrt((X(i,d) ^ 2) + (Y(i,d) ^ 2));
      % Solution
      S(i,d) = 1 / dist;
    end
    % Boundary
    SB = S(i,:); SB(SB > ub) = ub; SB(SB < lb) = lb; 
    S(i,:) = SB;
  end
  curve(t) = fitG;
  fprintf('\nGeneration %d Best (FOA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
FOA.sf = Sf; 
FOA.ff = sFeat;
FOA.nf = length(Sf); 
FOA.c  = curve; 
FOA.f  = feat;
FOA.l  = label;
end





