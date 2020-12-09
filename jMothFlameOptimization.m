%[2015]-"Moth-flame optimization algorithm: A novel nature-inspired
%heuristic paradigm"

% (9/12/2020)

function MFO = jMothFlameOptimization(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
b     = 1;     % constant

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'b'), b = opts.b; end 
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
fitG = inf;

curve = inf;
t = 1; 
while t <= max_Iter
  for i=1:N
    % Fitness
    fit(i) = fun(feat,label,(X(i,:) > thres),opts); 
    % Global best
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  if t == 1
    % Best flame
    [fitF, idx] = sort(fit,'ascend');
    flame       = X(idx,:); 
  else
    % Sort population
    XX        = [flame; X];
    FF        = [fitF, fit]; 
    [FF, idx] = sort(FF,'ascend'); 
    flame     = XX(idx(1:N),:);
    fitF      = FF(1:N);
  end
  % Flame update (3.14)
  flame_no = round(N - t * ((N - 1) / max_Iter));
  % Convergence constant, decreases linearly from -1 to -2 
  r = -1 + t * (-1 / max_Iter);
  for i = 1:N
    % Normal position update
    if i <= flame_no
      for d = 1:dim
        % Parameter T0, from r to 1
        T      = (r - 1) * rand() + 1;
        % Distance between flame & moth (3.13)
        dist   = abs(flame(i,d) - X(i,d));
        % Moth update (3.12)
        X(i,d) = dist * exp(b * T) * cos(2 * pi * T) + flame(i,d);
      end
    % Position update respect to best flames 
    else
      for d = 1:dim
        % Parameter T, from r to 1
        T      = (r - 1) * rand() + 1;
        % Distance between flame & moth (3.13)
        dist   = abs(flame(i,d) - X(i,d));
        % Moth update (3.12)
        X(i,d) = dist * exp(b * T) * cos(2 * pi * T) + ...
          flame(flame_no, d);
      end
    end
    % Boundary
    XB = X(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    X(i,:) = XB;
  end
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (MFO)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos((Xgb > thres) == 1); 
sFeat = feat(:,Sf);
% Store results
MFO.sf = Sf; 
MFO.ff = sFeat; 
MFO.nf = length(Sf);
MFO.c  = curve; 
MFO.f  = feat;
MFO.l  = label;
end





