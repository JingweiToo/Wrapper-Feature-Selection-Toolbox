%[2020]-"Generalized normal distribution optimization and its 
%applications in parameter extraction of photovoltaic models"

% (8/12/2020)

function GNDO = jGeneralizedNormalDistributionOptimization(...
  feat,label,opts)
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
    Xb   = X(i,:);
  end
end
% Pre
V = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Iteration
while t <= max_Iter
  % Compute mean position (22)
  M = mean(X,1);
  for i = 1:N
    alpha = rand();
    % [Local exploitation]
    if alpha > 0.5   
      % Random numbers
      a = rand(); 
      b = rand(); 
      for d = 1:dim
        % Compute mean (19)
        mu    = (1/3) * (X(i,d) + Xb(d) + M(d));
        % Compute standard deviation (20)
        delta = sqrt((1/3) * ((X(i,d) - mu) ^ 2 + ...
          (Xb(d) - mu) ^ 2 + (M(d) - mu) ^ 2));
        % Compute eta (21)
        lambda1 = rand();
        lambda2 = rand();
        if a <= b
          eta = sqrt(-1 * log(lambda1)) * cos(2 * pi * lambda2);
        else
          eta = sqrt(-1 * log(lambda1)) * cos(2 * pi * lambda2 + pi);
        end
        % Generate normal ditribution (18)
        V(i,d) = mu + delta * eta;
      end
    % [Global Exploitation] 
    else
      % Random three vectors but not i
      RN = randperm(N); RN(RN == i) = []; 
      p1 = RN(1); 
      p2 = RN(2);
      p3 = RN(3); 
      % Random beta
      beta = rand(); 
      % Normal random number: zero mean & unit variance
      lambda3 = randn();
      lambda4 = randn(); 
      % Get v1 (24)
      if fit(i) < fit(p1)
        v1 = X(i,:) - X(p1,:);
      else
        v1 = X(p1,:) - X(i,:);
      end
      % Get v2 (25)
      if fit(p2) < fit(p3)
        v2 = X(p2,:) - X(p3,:);
      else
        v2 = X(p3,:) - X(p2,:);
      end
      % Generate new position (23)
      for d = 1:dim
        V(i,d) = X(i,d) + beta * (abs(lambda3) * v1(d)) + ...
          (1 - beta) * (abs(lambda4) * v2(d));
      end
    end
    % Boundary
    XB = V(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    V(i,:) = XB;
  end 
  % Fitness
  for i = 1:N
    fitV = fun(feat,label,(V(i,:) > thres),opts);
    % Greedy selection (27)
    if fitV < fit(i)
      fit(i) = fitV; 
      X(i,:) = V(i,:);
    end
    % Best
    if fit(i) < fitG
      fitG = fit(i); 
      Xb   = X(i,:);
    end
  end
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (GNDO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim;
Sf    = Pos((Xb > thres) == 1); 
sFeat = feat(:,Sf); 
% Store results
GNDO.sf = Sf;
GNDO.ff = sFeat; 
GNDO.nf = length(Sf);
GNDO.c  = curve; 
GNDO.f  = feat; 
GNDO.l  = label;
end



