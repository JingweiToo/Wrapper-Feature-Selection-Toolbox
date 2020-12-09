%[2012]-"Flower pollination algorithm for global optimization"

% (9/12/2020)

function FPA = jFlowerPollinationAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
beta  = 1.5;    % levy component
P     = 0.8;    % switch probability

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'P'), P = opts.P; end 
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
% Compute fitness 
fit  = zeros(1,N); 
fitG = inf;
for i = 1:N
  fit(i) =fun(feat,label,(X(i,:) > thres),opts); 
  % Best flower
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
	for i = 1:N
    % Global pollination 
    if rand() < P
      % Levy distribution (2)
      L = jLevyDistribution(beta,dim); 
      for d = 1:dim
        % Global pollination (1)
        Xnew(i,d) = X(i,d) + L(d) * (X(i,d) - Xgb(d)); 
      end
    % Local pollination
    else
      % Different flower j, k in same species
      R   = randperm(N); 
      J   = R(1); 
      K   = R(2);
      % Epsilon [0 to 1]
      eps = rand();
      for d = 1:dim
        % Local pollination (3)
        Xnew(i,d) = X(i,d) + eps * (X(J,d) - X(K,d));
      end
    end
    % Check boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB; 
  end
  % Fitness
  for i = 1:N
    % Compute fitness
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Update if there is better solution
    if Fnew <= fit(i)
      X(i,:) = Xnew(i,:);
      fit(i) = Fnew;
    end
    % Best flower
    if fit(i) < fitG
      Xgb  = X(i,:); 
      fitG = fit(i);
    end
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (FPA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
FPA.sf = Sf; 
FPA.ff = sFeat;
FPA.nf = length(Sf);
FPA.c  = curve;
FPA.f  = feat; 
FPA.l  = label;
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




