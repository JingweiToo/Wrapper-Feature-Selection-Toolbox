% [2009]-"GSA: A gravitational search algorithm"

% (9/12/2020)

function GSA = jGravitationalSearchAlgorithm(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
G0    = 100;   % initial gravitational constant
alpha = 20;    % cosntant

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'G0'), G0 = opts.G0; end  
if isfield(opts,'alpha'), alpha = opts.alpha; end 
if isfield(opts,'thres'), thres = opts.thres; end

% Objective function
fun = @jFitnessFunction;
% Number of dimensions
dim = size(feat,2); 
% Initial population
X   = zeros(N,dim);
V   = zeros(N,dim); 
for i = 1:N
  for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
  end
end  
% Pre 
fit  = zeros(1,N);
fitG = inf;

curve = inf;
t= 1 ;
% Iteration
while t <= max_Iter
  for i = 1:N
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts);
    % Global best 
    if fit(i) < fitG
      Xgb  = X(i,:); 
      fitG = fit(i);
    end
  end
	% The best & the worst fitness (17-18)
  best  = min(fit); 
  worst = max(fit);
  % Normalization mass (15)
  mass  = (fit - worst) ./ (best - worst); 
  % Compute inertia mass (16)
  M     = mass ./ sum(mass); 
  % Update gravitaty constant (28)
  G     = G0 * exp(-alpha * (t / max_Iter)); 
  % Kbest linearly decreases from N to 1 
  Kbest = round(N - (N - 1) * (t / max_Iter)); 
  % Sort mass in descending order
  [~, idx_M] = sort(M,'descend'); 
  E = zeros(N,dim);
  for i = 1:N
  	for ii = 1:Kbest
      j = idx_M(ii);
      if j ~= i
        % Euclidean distance (8)
        R = sqrt(sum((X(i,:) - X(j,:)) .^ 2)); 
        for d = 1:dim
          % Note that Mp(i)/M(i)=1 (7,9)
          E(i,d) = E(i,d) + rand() * M(j) * ...
            ((X(j,d) - X(i,d)) / (R + eps)); 
        end
      end
    end
  end
  % Search agent update
  for i = 1:N
    for d = 1:dim
      % Acceleration: Note Mii(t) ~1 (10)
      Acce   = E(i,d) * G;  
      % Velocity update (11)
      V(i,d) = rand() * V(i,d) + Acce; 
      % Position update (12)
      X(i,d) = X(i,d) + V(i,d);
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (GSA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1);
sFeat = feat(:,Sf);
% Store results
GSA.sf = Sf; 
GSA.ff = sFeat; 
GSA.nf = length(Sf);
GSA.c  = curve; 
GSA.f  = feat; 
GSA.l  = label;
end





