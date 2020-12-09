%[2009]-"Cuckoo search via Levy flights" 

% (9/12/2020)

function CS = jCuckooSearchAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
Pa    = 0.25;   % discovery rate
alpha = 1;      % constant
beta  = 1.5;    % levy component

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'Pa'), Pa = opts.Pa; end 
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'beta'), beta = opts.beta; end 
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
  % Best cuckoo nest
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
% Iterations
while t <= max_Iter
  % {1} Random walk/Levy flight phase
  for i = 1:N
    % Levy distribution
    L = jLevyDistribution(beta,dim);
    for d = 1:dim
      % Levy flight (1)
      Xnew(i,d) = X(i,d) + alpha * L(d) * (X(i,d) - Xgb(d));
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    Xnew(i,:) = XB; 
  end
  % Fintess
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts); 
    % Greedy selection
    if Fnew <= fit(i)
      fit(i) = Fnew;
      X(i,:) = Xnew(i,:);
    end
  end
  % {2} Discovery and abandon worse nests phase
  Xj = X(randperm(N),:); 
  Xk = X(randperm(N),:);
  for i = 1:N 
    Xnew(i, :) = X(i,:);
    r          = rand();
    for d = 1:dim
      % A fraction of worse nest is discovered with a probability
      if rand() < Pa
        Xnew(i,d) = X(i,d) + r * (Xj(i,d) - Xk(i,d));
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb; 
    Xnew(i,:) = XB; 
  end
  % Fitness
  for i = 1:N
    % Fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts); 
    % Greedy selection
    if Fnew <= fit(i)
      fit(i) = Fnew;
      X(i,:) = Xnew(i,:);
    end
    % Best cuckoo
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (CS)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
CS.sf = Sf; 
CS.ff = sFeat; 
CS.nf = length(Sf); 
CS.c  = curve;
CS.f  = feat;
CS.l  = label;
end


%// Levy Flight //
function LF = jLevyDistribution(beta,dim)
% Sigma 
nume  = gamma(1 + beta) * sin(pi * beta / 2); 
deno  = gamma((1 + beta) / 2) * beta * 2 ^ ((beta - 1) / 2);
sigma = (nume / deno) ^ (1 / beta); 
% Parameter u & v 
u = randn(1,dim) * sigma; 
v = randn(1,dim);
% Step 
step = u ./ abs(v) .^ (1 / beta);
LF   = 0.01 * step;
end


