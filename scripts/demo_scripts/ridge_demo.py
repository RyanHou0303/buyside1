import numpy as np
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(10, 5.5))


ax.semilogx(lambdas, train_mse, label='Training')
ax.semilogx(lambdas, test_mse, label='Test')
ax.axvline(opt_lambda, ls='--', c='orange', label=f'Optimal $\\lambda$: {opt_lambda:.2f}')


ax.invert_xaxis()


ax.set(title='Error vs Model Complexity', xlabel='Regularization Strength $\\lambda$', ylabel='MSE')
ax.grid(True, ls='--', alpha=0.4)
ax.legend()

plt.show()
