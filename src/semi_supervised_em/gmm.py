import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 2: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples
    
    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    def initialize_mu_sigma(X, K):
        n_samples, n_features = X.shape
        
        # Randomly shuffle the indices and assign each example to one of K groups
        indices = np.random.permutation(n_samples)
        groups = np.array_split(indices, K)  # Split the indices into K groups

        mu = np.zeros((K, n_features))  # Means for each group
        sigma = []  # Covariance matrices for each group

        for k in range(K):
            # Extract the data points in the k-th group
            group_data = X[groups[k]]
            
            # Calculate the sample mean for this group
            mu[k] = np.mean(group_data, axis=0)
            
            # Calculate the sample covariance matrix for this group
            group_cov = np.cov(group_data, rowvar=False)  # Set rowvar=False for column-wise data
            sigma.append(group_cov)
        return mu, sigma
    mu, sigma = initialize_mu_sigma(x, K)
    phi = np.ones(K)/K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((len(x), K))/K
    # *** END CODE HERE ***

    if is_semi_supervised:
        print('semi')
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    n=len(x)
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)

def log_likelihood(X, mu, sigma, phi):
    n_samples, n_features = X.shape
    K = len(phi)  # Number of components
    ll = 0.0  # Log-likelihood
    #ll = sum_x[log(sum_z[p(x|z) * p(z)])]
    # Loop over each data point
    for i in range(n_samples):
    # Calculate the weighted sum of Gaussian densities for each component
        weighted_sum = 0.0
        for k in range(K):
            diff = X[i] - mu[k]
            sigma_inv = np.linalg.inv(sigma[k])  # Inverse of the covariance matrix
            sigma_det = np.linalg.det(sigma[k])  # Determinant of the covariance matrix
                    
            # Calculate the Mahalanobis distance (quadratic form)
            m_dist = np.dot(diff, np.dot(sigma_inv, diff))
                    
            # Compute the Gaussian PDF for component k at point x_i
            gauss_pdf = np.exp(-0.5 * m_dist) / np.sqrt((2 * np.pi)**n_features * sigma_det)
                    
            # Add the weighted Gaussian contribution
            weighted_sum += phi[k] * gauss_pdf
                
            # Take the log of the weighted sum and accumulate to log-likelihood
        ll += np.log(weighted_sum)

    return ll

#ll = log_likelihood_semi(x,x_tilde, z, mu, w_tilde, sigma, phi, alpha)
def log_likelihood_semi(X,x_tilde, z, mu, sigma, phi, alpha):
    ll = log_likelihood(X, mu, sigma, phi)  
    n_tilde_samples, n_tilde_features = x_tilde.shape
    # Loop over each data point
    ll_two = 0.0
    for i in range(n_tilde_samples):
        k = int(z[i])  
        diff = x_tilde[i] - mu[k]

        sigma_inv = np.linalg.inv(sigma[k])  # Inverse of the covariance matrix
        sigma_det = np.linalg.det(sigma[k])  # Determinant of the covariance matrix
                    
        m_dist = np.dot(diff, np.dot(sigma_inv, diff))
                    
        # Compute the Gaussian PDF for component k at point x_tilde[i]
        gauss_pdf = np.exp(-0.5 * m_dist) / np.sqrt((2 * np.pi) ** n_tilde_features * sigma_det)
        
        ll_two +=  np.log(phi[k] * gauss_pdf)
    return ll+alpha*ll_two

def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim).

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    n_samples=x.shape[0]
    n_features=x.shape[1]
    #print(x.shape[1])
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        #pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        prev_ll=ll
        for i in range(n_samples):
            #print(np.dot(sigma_inv, (x[i]-mu[0])))
            #x_i_const = sum([np.exp(-0.5*np.dot((x[i]-mu[l]), np.dot(np.linalg.inv(sigma[l]), (x[i]-mu[l]))))*phi[l] for l in range(len(mu))])
            #print('constant denom', x_i_const)
            for j in range(K):
                diff = x[i] - mu[j]
                sigma_inv = np.linalg.inv(sigma[j])  # Inverse of the covariance matrix
                m_dist = np.dot(diff, np.dot(sigma_inv, diff))
                gaussian_pdf = np.exp(-0.5 * m_dist) / np.sqrt(((2 * np.pi) ** n_features) * np.linalg.det(sigma[j]))
            
                w[i, j] = phi[j] * gaussian_pdf
                #print("w", i, j, ":", w[i, j])
        for i in range(n_samples):
            x_i_const = np.sum(w[i, :])  # Normalization constant (sum over all components)
            if x_i_const > 0:
                w[i, :] /= x_i_const  # Normalize the weights

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            phi[j] = np.mean([w[i][j] for i in range(n_samples)])
            mu[j]=np.sum([w[i][j]*x[i] for i in range(n_samples)])*(1/np.sum([w[i][j] for i in range(n_samples)]))
            sigma[j]=np.sum([w[i][j]*np.outer(x[i]-mu[j], x[i]-mu[j]) for i in range(n_samples)], axis=0)*(1/np.sum([w[i][j] for i in range(n_samples)]))
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        ll =  log_likelihood(x, mu, sigma, phi)
        it+=1
        print(f"Iteration {it}: Log-Likelihood = {ll}")
        # *** END CODE HERE ***
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """
    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    n_samples, n_features=x.shape
    n_tilde = x_tilde.shape[0]
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    it = 0
    ll = prev_ll = None

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # (1) E-step: Update your estimates in w

        prev_ll = ll

        for i in range(n_samples):
            for j in range(K):
                diff = x[i] - mu[j]
                sigma_inv = np.linalg.inv(sigma[j])  # Inverse of the covariance matrix
                m_dist = np.dot(diff, np.dot(sigma_inv, diff))
                gaussian_pdf = np.exp(-0.5 * m_dist) / np.sqrt(((2 * np.pi) ** n_features) * np.linalg.det(sigma[j]))
                w[i, j] = phi[j] * gaussian_pdf
                
        for i in range(n_samples):
            x_i_const = np.sum(w[i, :]) 
            if x_i_const > 0:
                w[i, :] /= x_i_const  

        w = w / w.sum(axis=1, keepdims=True)
        #Update W_tilde
        w_tilde = np.zeros((n_tilde, K))
        for i in range(n_tilde):
            w_tilde[i, int(z[i])] = 1  # Set responsibility for the true label

        w_tilde = w_tilde / np.sum(w_tilde, axis=1, keepdims=True)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        for j in range(K):
            mu[j] = (w[:, j].dot(x) + alpha * w_tilde[:, j].dot(x_tilde)) / ((w[:, j].sum() + alpha * w_tilde[:, j].sum()))
            sigma[j] = (np.sum([w[i][j] * np.outer(x[i] - mu[j], x[i] - mu[j]) for i in range(n_samples)], axis=0) + alpha * np.sum([w_tilde[i][j] * np.outer(x_tilde[i] - mu[j], x_tilde[i] - mu[j]) for i in range(n_tilde)], axis=0)) / (np.sum([w[i][j] for i in range(n_samples)]) + alpha * np.sum([w_tilde[i][j] for i in range(n_tilde)]))
        phi = (w.sum(axis=0) + alpha * w_tilde.sum(axis=0))
        phi = phi / phi.sum()

        # compute log likelihood
        
        # (3) Compute the log-likelihood of the data to check for convergence.
        ll = log_likelihood_semi(x, x_tilde, z, mu, sigma, phi, alpha)#np.sum(np.log(p_x)) + alpha * np.sum(np.log(p_x_z))
        it += 1
        print('Iter {}, Likelihood {}'.format(it, ll))

    return w


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    #plt.show()
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        #main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.

        main(is_semi_supervised=True, trial_num=t)

        # *** END CODE HERE ***
