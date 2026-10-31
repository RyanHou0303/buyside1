import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

#first demo is about ridge in data that we generated
np.random.seed(42)
n=1000
p=100

beta_true = np.zeros(p)
beta_true[:10]=0.1
print(beta_true)

X_train = np.random.randn(n,p)
y_train = X_train@beta_true+np.random.randn(n)

X_test=np.random.randn(n,p)
y_test = X_test@beta_true+np.random.randn(n)

lambdas=np.logspace(-1,5,200)
#the formula of ridge is (X.t@X+lambda)^-1@XTy
XtX= X_train.T@X_train
Xty= X_train.T@y_train

eigvals,Q = np.linalg.eigh(XtX)#get Q@diag(eigenvals)@Q.T, a quicker way
Qt_Xty = Q.T@Xty


train_mse = []
test_mse = []
for lam in lambdas:
    beta_hat = Q @ (Qt_Xty / (eigvals + lam))#speed up
    train_mse.append(np.mean((y_train-X_train@beta_hat)**2))
    test_mse.append(np.mean((y_test-X_test@beta_hat)**2))

train_mse = np.array(train_mse)
test_mse = np.array(test_mse)

opt_idx = np.argmin(test_mse)
opt_lambda = lambdas[opt_idx]
beta_ridge_opt = Q @ (Qt_Xty / (eigvals + opt_lambda))
#if we consider lasso:
lambda_lasso = np.logspace(0,3,100)
lasso_train_mse,lasso_test_mse=[],[]

for lam in lambda_lasso:
    model = Lasso(alpha=lam/(2*n),fit_intercept=False)
    model.fit(X_train,y_train)

    lasso_train_mse.append(np.mean((y_train - X_train @ model.coef_)**2))
    lasso_test_mse.append(np.mean((y_test - X_test @ model.coef_)**2))

opt_idx_lasso = np.argmin(lasso_test_mse)
opt_lambda_lasso = lambda_lasso[opt_idx_lasso]

best_lasso = Lasso(alpha=opt_lambda_lasso / (2 * n), fit_intercept=False).fit(X_train, y_train)
beta_lasso_opt = best_lasso.coef_

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))


ax1.semilogx(lambdas, train_mse, label='Training')
ax1.semilogx(lambdas, test_mse, label='Test')
ax1.axvline(opt_lambda, ls='--', c='orange', label=f'Optimal $\\lambda$: {opt_lambda:.2f}')
ax1.invert_xaxis()
ax1.set(title='Ridge: Error vs $\\lambda$', xlabel='Ridge $\\lambda$ (Inverted)', ylabel='MSE')
ax1.grid(True, ls='--', alpha=0.4)
ax1.legend()


ax2.semilogx(lambda_lasso, lasso_train_mse, label='Training')
ax2.semilogx(lambda_lasso, lasso_test_mse, label='Test')
ax2.axvline(opt_lambda_lasso, ls='--', c='orange', label=f'Optimal $\\lambda$: {opt_lambda_lasso:.2f}')
ax2.invert_xaxis()
ax2.set(title='Lasso: Error vs $\\lambda$', xlabel='Lasso $\\lambda$ (Inverted)', ylabel='MSE')
ax2.grid(True, ls='--', alpha=0.4)
ax2.legend()

ax3.plot(beta_true, 'k-', lw=2, label='True $\\beta$ (First 10 = 0.1)')
ax3.plot(beta_ridge_opt, 'b.', alpha=0.6, label='Optimal Ridge $\\hat{\\beta}$')
ax3.plot(beta_lasso_opt, 'rx', alpha=0.8, label='Optimal Lasso $\\hat{\\beta}$')
ax3.set(title='Recovered Weights $\\hat{\\beta}$ Comparison', xlabel='Feature Index (0~99)', ylabel='Weight Value')
ax3.grid(True, ls='--', alpha=0.4)
ax3.legend()

plt.tight_layout()
plt.show()
