%[2020]-"Manta ray foraging optimization: An effective bio-inspired
%optimizer for engineering applications"

% (8/12/2020)

function MRFO = jMantaRayForagingOptimization(feat,label,opts)
% Parameters
lb    = 0;
ub    = 1; 
thres = 0.5; 
S     = 2;     % somersault factor 

if isfield(opts,'N'), N = opts.N; end
if isfield(opts,'T'), max_Iter = opts.T; end
if isfield(opts,'S'), S = opts.S; end 
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
    fitG  = fit(i);
    Xbest = X(i,:);
  end
end
% Pre
Xnew  = zeros(N,dim);

curve = zeros(1,max_Iter);
curve(1) = fitG;
t = 2; 
% Iteration
while t <= max_Iter
  for i = 1:N
    % [Cyclone foraging]
    if rand() < 0.5
      if t / max_Iter < rand() 
        % Compute beta (5)
        r1    = rand();  
        beta  = 2 * exp(r1 * ((max_Iter - t + 1) / max_Iter)) * ...
          (sin(2 * pi * r1));
        for d = 1:dim
          % Create random solution (6) 
          Xrand = lb + rand() * (ub - lb);
          % First manta ray follow best food (7)
          if i == 1
            Xnew(i,d) = Xrand + rand() * (Xrand - X(i,d)) + ...
              beta * (Xrand - X(i,d));
          % Followers follew the front manta ray (7)
          else
            Xnew(i,d) = Xrand + rand() * (X(i-1,d) - X(i,d)) + ...
              beta * (Xrand - X(i,d));
          end
        end
      else
        % Compute beta (5)
        r1   = rand(); 
        beta = 2 * exp(r1 * ((max_Iter - t + 1) / max_Iter)) * ...
          (sin(2 * pi * r1));
        for d = 1:dim
          % First manta ray follow best food (4)
          if i == 1
            Xnew(i,d) = Xbest(d) + rand() * (Xbest(d) - X(i,d)) + ...
              beta * (Xbest(d) - X(i,d));
          % Followers follow the front manta ray (4)
          else
            Xnew(i,d) = Xbest(d) + rand() * (X(i-1,d) - X(i,d)) + ...
              beta * (Xbest(d) - X(i,d));
          end          
        end
      end
    % [Chain foraging] 
    else
      for d = 1:dim
        % Compute alpha (2)
        r     = rand(); 
        alpha = 2 * r * sqrt(abs(log(r)));
        % First manta ray follow best food (1)
        if i == 1
          Xnew(i,d) = X(i,d) + rand() * (Xbest(d) - X(i,d)) + ...
            alpha * (Xbest(d) - X(i,d));
        % Followers follew the front manta ray (1)
        else
          Xnew(i,d) = X(i,d) + rand() * (X(i-1,d) - X(i,d)) + ...
            alpha * (Xbest(d) - X(i,d));
        end
      end
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Greedy selection
    if Fnew < fit(i)
      fit(i) = Fnew; 
      X(i,:) = Xnew(i,:);
    end
    % Update best
    if fit(i) < fitG
      fitG  = fit(i);
      Xbest = X(i,:);
    end
  end
  % [Somersault foraging] 
  for i = 1:N
    % Manta ray update (8)
    r2 = rand(); 
    r3 = rand();
    for d = 1:dim 
      Xnew(i,d) = X(i,d) + S * (r2 * Xbest(d) - r3 * X(i,d));
    end
    % Boundary
    XB = Xnew(i,:); XB(XB > ub) = ub; XB(XB < lb) = lb;
    Xnew(i,:) = XB;
  end
  % Fitness
  for i = 1:N
    Fnew = fun(feat,label,(Xnew(i,:) > thres),opts);
    % Greedy selection
    if Fnew < fit(i)
      fit(i) = Fnew; 
      X(i,:) = Xnew(i,:);
    end
    % Update best
    if fit(i) < fitG
      fitG  = fit(i);
      Xbest = X(i,:);
    end
  end 
  curve(t) = fitG; 
  fprintf('\nIteration %d Best (MRFO)= %f',t,curve(t))
  t = t + 1;
end
% Select features based on selected index
Pos   = 1:dim; 
Sf    = Pos((Xbest > thres) == 1);
sFeat = feat(:,Sf); 
% Store results
MRFO.sf = Sf; 
MRFO.ff = sFeat; 
MRFO.nf = length(Sf);
MRFO.c  = curve;
MRFO.f  = feat; 
MRFO.l  = label;
end



