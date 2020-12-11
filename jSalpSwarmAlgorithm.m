%[2017]-"Salp swarm algorithm: A bio-inspired optimizer for engineering
%design problems"

% (8/12/2020)

function SSA = jSalpSwarmAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5;

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
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
fitF = inf;

curve = inf; 
t = 1; 
% Iteration
while t <= max_Iter
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Best food update
    if fit(i) < fitF
      Xf   = X(i,:);
      fitF = fit(i); 
    end
  end
	% Compute coefficient, c1 (3.2)
	c1 = 2 * exp(-(4 * t / max_Iter) ^ 2);
	for i = 1:N
    % Leader update
    if i == 1
      for d = 1:dim
        % Coefficient c2 & c3 [0~1]
        c2 = rand(); 
        c3 = rand();
      	% Leader update (3.1)
        if c3 >= 0.5 
          X(i,d) = Xf(d) + c1 * ((ub - lb) * c2 + lb);
        else
          X(i,d) = Xf(d) - c1 * ((ub - lb) * c2 + lb);
        end
      end
    % Salp update
    elseif i >= 2
      for d = 1:dim
        % Salp update by following front salp (3.4)
        X(i,d) = (X(i,d) + X(i-1,d)) / 2;
      end
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end
  curve(t) = fitF; 
  fprintf('\nIteration %d Best (SSA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim; 
Sf    = Pos((Xf > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
SSA.sf = Sf;
SSA.ff = sFeat; 
SSA.nf = length(Sf);
SSA.c  = curve;
SSA.f  = feat; 
SSA.l  = label;
end





