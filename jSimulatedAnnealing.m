%[1983]-"Optimization by Simulated Annealing"

% (9/12/2020)

function SA = jSimulatedAnnealing(feat,label,opts)
% Parameters 
c  = 0.93;  % cooling rate
T0 = 100;   % initial temperature

if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'c'), c = opts.c; end 
if isfield(opts,'T0'), T0 = opts.T0; end 

% Objective function
fun = @jFitnessFunction; 
% Number of dimensions
dim = size(feat,2); 
% Initial 
X   = jInitialization(1,dim);
% Fitness 
fit = fun(feat,label,X,opts); 
% Initial best
Xgb  = X; 
fitG = fit;
% Pre
curve = zeros(1,max_Iter);
t = 2;
% Iterations
while t <= max_Iter
	% Probabilty of swap, insert, flip & eliminate
  prob = randi([1,4]);
  % Swap operation
  if prob == 1
    Xnew  = X;
    % Find index with bit '0' & '1'
    bit0  = find(X == 0);
    bit1  = find(X == 1);
    len_0 = length(bit0); 
    len_1 = length(bit1);
    % Solve issue with missing bit '0' or '1'
    if len_0 ~= 0 && len_1 ~= 0
      % Get one random index from x1 & x2
      ind0 = randi([1,len_0]); 
      ind1 = randi([1,len_1]);
      % Swap between two index
      Xnew(bit0(ind0)) = 1; 
      Xnew(bit1(ind1)) = 0;
    end
 
  % Insert operation
  elseif prob == 2
    Xnew  = X;
    % Find index with zero
    bit0  = find(X == 0);
    len_0 = length(bit0); 
    % Solve problem when all index are '1'
    if len_0 ~= 0 
      ind = randi([1,len_0]);
      % Add one feature
      Xnew(bit0(ind)) = 1;
    end
  
  % Eliminate operation
  elseif prob == 3
    Xnew  = X;
    % Find index with one
    bit1  = find(X == 1);
    len_1 = length(bit1);
    % Solve problem when all index are '0'
    if len_1 ~= 0
      ind = randi([1,len_1]);
      % Remove one feature
      Xnew(bit1(ind)) = 0;
    end
  
  % Flip operation
  elseif prob == 4
    Xnew = X;
    % Flip all variables
    for d = 1:dim
      Xnew(d) = 1 - Xnew(d);
    end
  end
  
  % Fitness
  Fnew = fun(feat,label,Xnew,opts);
  % Global best update 
  if Fnew <= fitG
    Xgb  = Xnew; 
    fitG = Fnew; 
    X    = Xnew; 
  % Accept worst solution with probability
  else 
    % Delta energy 
    delta = Fnew - fitG;
    % Boltzmann Probility 
    P     = exp(-delta / T0);
    if rand() <= P
      X = Xnew; 
    end
  end 
  % Temperature update
  T0 = c * T0; 
  % Save
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (SA)= %f',t,curve(t))
  t = t + 1;
end
% Select features
Pos   = 1:dim;
Sf    = Pos(Xgb == 1); 
sFeat = feat(:,Sf);
% Store results
SA.sf = Sf; 
SA.ff = sFeat; 
SA.nf = length(Sf);
SA.c  = curve; 
SA.f  = feat;
SA.l  = label;
end


function X = jInitialization(N,dim)
% Initialize X vectors
X = zeros(N,dim);
for i = 1:N
  for d = 1:dim 
    if rand() > 0.5
      X(i,d) = 1;
    end
  end
end
end



