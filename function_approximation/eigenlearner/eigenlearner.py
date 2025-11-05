import numpy as np

class EigenLearner:

    def __init__(self, env, dataset, lambd=1.0):

        self.env = env
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.lambd = lambd

        ### useful objects to have
        r = self.env.rewards
        p = self.env.transition_probs

        # uniform random policy
        pi = np.ones((self.env.num_states, self.env.num_actions)) / self.env.num_actions
        self.P = (p * pi[..., None]).sum(1)       # P as defined in the DR paper.
        self.R = np.diag(np.exp(-r / lambd))             # R as defined in the DR paper.
        self.R_inv = np.diag(np.exp(r / lambd))         # R^(-1)
        self.R_inv_sqrt = np.diag(np.exp(r / lambd / 2))         # R^(-1/2)

        
    def init_learn(self):
        self.compute_matrix()
        self.compute_top_eigvec()
        self.eigvec = np.zeros((self.env.num_states,))

        self.cos_sims = []

    def compute_matrix(self):
        """
        Compute each matrix        
        """
        pass
    
    def compute_top_eigvec(self):
        """
        Matrix-specific computation of ground-truth eigvec
        """
        self.true_eigvec = None

    def update(self, step_size=0.001):
        """
        Matrix-specifc update to self.eigvec
        """
        pass


    def cos_sim(self,):
        """
        Compute cosine similarity between learned eigenvec and ground-truth eigenvec
        """
        return self.eigvec @ self.true_eigvec / (np.linalg.norm(self.eigvec) * np.linalg.norm(self.true_eigvec))
    
    def sym(self, M):
        """
        Return symmetrized version of matrix as (M + M.T) / 2
        """
        return (M + M.T) / 2


    def learn(self, n_epochs=10000, step_size=0.001):

        for i in range(n_epochs):
            self.update(step_size=step_size)

            # print(self.eigvec)

            # log cosine similarity
            self.cos_sims.append(self.cos_sim())

            if (i + 1) % 100 == 0:
                print(self.cos_sims[-1])