function model = train( J, X, Y, parameter)
%% parameters
lambda     = parameter.lambda;
lambda2    = parameter.lambda2;
lambda3    = parameter.lambda3;
lambda4    = parameter.lambda4;
lambda5    = parameter.lambda5;
lambda6    = parameter.lambda6;
alpha      = parameter.alpha;
rho        = parameter.rho;
mu         = parameter.mu;
maxMu      = parameter.maxMu;
epsilon    = parameter.epsilon;
maxIter    = parameter.maxIter;

%% initialization
[~,num_dim]  = size(X);
[~,num_class]  = size(Y);

W = eye(num_dim, num_class);
B = W; % Q
Lambda = W - B;
W_1 = W;

% label correlation C
C = zeros(num_class, num_class);
C_1 = C;

% feature correlation P
[Wd] = UpdateP(J, X, Y);
Ld = Wd*Wd'; % M

% representative
paraDc = 0.15;
[Q, ~] = InstanceRepresentativeness(X, paraDc);
Ldd = Q;

Ymis = Y;
Ymis(Ymis==-1)=0;
YTY = Ymis'*Ymis;
XTX = X'*X;

Lipw1 = norm(XTX)^2;
Lipw1 = Lipw1 + 2*norm(2*lambda4*Ld)^2*norm(YTY)^2;
Lipw1 = Lipw1 + norm(lambda5*(X - Ldd*X)'*(X - Ldd*X))^2;

Lipc1 = norm(lambda*Y'*Y)^2;

iter = 1;
bk = 1;
bk_1 = 1;

while iter <= maxIter
    %% Lip
    L = C*C';
    Lipw2 = Lipw1 + mu^2;
    Lipw1 = Lipw1 + 2*norm(2*lambda3*XTX)^2*norm(L)^2;
    Lipw = sqrt(5*Lipw2);
    
    Lipc2 = Lipc1 + norm(lambda3*W'*X'*X*W)^2;
    Lipc = sqrt(2*Lipc2);
    
    %% update W
    W_k  = W + (bk_1 - 1)/bk * (W - W_1);
    Gw_x_k = W_k - 1/Lipw * gradientOfW(J,XTX,YTY,W,L,Ld,Ldd,B,Lambda,lambda3,lambda4,lambda5,mu,X,Y);
    W_1 = W;
    W = softthres(Gw_x_k, (lambda2*(1-alpha))/Lipw);
    
    %% update C
    C_k  = C + (bk_1 - 1)/bk * (C - C_1);
    Gc_x_k = C_k - 1/Lipc * gradientOfC(W,C,lambda,lambda3,X,Y);
    C_1 = C;
    C = softthres(Gc_x_k, lambda6/Lipc);
    
    %% update B
    [U, Sigma, V] = svd(W + Lambda/mu,'econ');
    B = U * softthres(Sigma,(lambda2*alpha)/mu) * V';
    
    %% update Lambda
    % Lmabda
    Lambda = Lambda + mu*(W - B);
    % mu
    mu = min(maxMu, mu*rho);
    % b
    bk_1 = bk;
    bk = (1 + sqrt(4*bk^2 + 1))/2;
    
    %% stop conditions
    if norm(W - B, 'inf') < epsilon
        break;
    end
    iter=iter+1;
end

model.W = W;
model.B = B;
model.Lambda = Lambda;
model.L = L;
model.Ld = Ld;
model.Ldd = Ldd;
model.YTY = YTY;
model.Lipw = Lipw;
model.iter = iter;
end

%% soft thresholding operator
function Ws = softthres(W,lambda)
Ws = max(W-lambda,0) - max(-W-lambda,0);
end
%% gradient W
function gradient = gradientOfW(J,XTX,YTY,W,L,Ld,Ldd,B,Lambda,lambda3,lambda4,lambda5,mu,X,Y)
gradient = X'*(J.*(X*W - Y));
gradient = gradient + 2*lambda3*XTX*W*L + 2*lambda4*Ld*W*YTY;
gradient = gradient + mu * (W - B) + Lambda;
gradient = gradient + lambda5*(X - Ldd*X)'*(X - Ldd*X)*W;
end
%% gradient C
function gradient = gradientOfC(W,C,lambda1,lambda3,X,Y)
XTX = X'*X;
gradient = lambda1*Y'*(Y*C-Y);
gradient = gradient + lambda3*W'*XTX*W*C;

end