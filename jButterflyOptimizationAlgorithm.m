%[2018]-"Butterfly optimization algorithm: a novel approach for global
%optimization"

% (9/12/2020)

function BOA = jButterflyOptimizationAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
c     = 0.01;   % modular modality
p     = 0.8;    % switch probability

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'c'), c = opts.c; end 
if isfield(opts,'p'), p = opts.p; end 
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
Xnew = zeros(N,dim);
fitG = inf; 
fit  = zeros(1,N);

curve = inf;
t = 1; 
% Iterations
while t <= max_Iter
  % Fitness 
  for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Global update
    if fit(i) < fitG
      fitG = fit(i); 
      Xgb  = X(i,:);
    end
  end 
  % Power component, increase from 0.1 to 0.3
  a = 0.1 + 0.2 * (t / max_Iter);
  for i = 1:N
    % Compute fragrance (1)
    f = c * (fit(i) ^ a);
    % Random number in [0,1]
    r = rand();
    if r < p
      r1 = rand();
      for d = 1:dim
        % Move toward best butterfly (2)
        Xnew(i,d) = X(i,d) + ((r1 ^ 2) * Xgb(d) - X(i,d)) * f;
      end
    else
      % Random select two butterfly
      R  = randperm(N); 
      J  = R(1); 
      K  = R(2);
      r2 = rand();
      for d = 1:dim
        % Move randomly (3)
        Xnew(i,d) = X(i,d) + ((r2 ^ 2) * X(J,d) - X(K,d)) * f;
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Replace
  X = Xnew;
  % Save
  curve(t) = fitG;
  fprintf('\nIteration %d Best (BOA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
BOA.sf = Sf; 
BOA.ff = sFeat; 
BOA.nf = length(Sf); 
BOA.c  = curve; 
BOA.f  = feat; 
BOA.l  = label;
end




