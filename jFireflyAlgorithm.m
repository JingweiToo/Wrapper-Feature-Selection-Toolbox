%[2010]-"Firefly algorithm,stochastic test functions and design 
%optimization" 

% (9/12/2020)

function FA = jFireflyAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
alpha = 1;      % constant
beta0 = 1;      % light amplitude
gamma = 1;      % absorbtion coefficient
theta = 0.97;   % control alpha

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'beta0'), beta0 = opts.beta0; end 
if isfield(opts,'gamma'), gamma = opts.gamma; end 
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'theta'), theta = opts.theta; end 
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
  % Best solution
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end
% Pre
curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Generation
while t <= max_Iter 
  % Alpha update
  alpha      = alpha * theta; 
  % Rank firefly based on their light intensity
  [fit, idx] = sort(fit,'ascend');
  X          = X(idx,:);
	for i = 1:N
    % The attractiveness parameter
    for j = 1:N
      % Update moves if firefly j brighter than firefly i
      if fit(i) > fit(j) 
        % Compute Euclidean distance 
        r    = sqrt(sum((X(i,:) - X(j,:)) .^ 2));
        % Beta (2)
        beta = beta0 * exp(-gamma * r ^ 2); 
        for d = 1:dim
          % Update position (3)
          eps    = rand() - 0.5;
          X(i,d) = X(i,d) + beta * (X(j,d) - X(i,d)) + alpha * eps; 
        end
        % Boundary 
        XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
        X(i,:) = XB;
        % Fitness 
        fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
        % Update global best firefly
        if fit(i) < fitG
          fitG = fit(i); 
          Xgb  = X(i,:);
        end
      end
    end
  end
  curve(t) = fitG;
  fprintf('\nGeneration %d Best (FA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
FA.sf = Sf; 
FA.ff = sFeat; 
FA.nf = length(Sf);
FA.c  = curve; 
FA.f  = feat; 
FA.l  = label;
end





