import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_device():
    '''
    Return a torch.device object. Returns a CUDA device if it is available and
    a CPU device otherwise.
    '''
    if torch.cuda.is_available():
        print('device is CUDA')
        return torch.device('cuda')
    else:
        print('device is CPU')
        return torch.device('cpu')

def show_plot(reward_log, folder="pics", algorithm="PPO", a=0.05):           
    try: 
        running_avg=0
        sum_reward=0
        i=0
        plt.clf()
        for plot_data in reward_log:
            sum_reward+=plot_data[1]
            i+=1
            if(i<10):
                running_avg=sum_reward/i
            else:
                running_avg=running_avg*(1-a)+plot_data[1]*a
            plt.plot(plot_data[0], plot_data[1], '.', color='r')
            plt.plot(plot_data[0], running_avg, '.', color='b')
        plt.savefig(f"{folder}/{algorithm}_reward_plot.png")
        plt.show()
    except Exception as e:
        print(str(e))

        
def apply_update(parameterized_fun, update):
    '''
    Add update to the weights of parameterized_fun

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator to be updated

    update : torch.FloatTensor
        a flattened version of the update to be applied
    '''

    n = 0

    for param in parameterized_fun.parameters():
        numel = param.numel()
        param_update = update[n:n + numel].view(param.size())
        param.data += param_update
        n += numel
        

def flatten(vecs):
    '''
    Return an unrolled, concatenated copy of vecs

    Parameters
    ----------
    vecs : list
        a list of Pytorch Tensor objects

    Returns
    -------
    flattened : torch.FloatTensor
        the flattened version of vecs
    '''

    flattened = torch.cat([v.view(-1) for v in vecs])

    return flattened

def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    '''
    Return a flattened view of the gradients of functional_output w.r.t. inputs

    Parameters
    ----------
    functional_output : torch.FloatTensor
        The output of the function for which the gradient is to be calculated

    inputs : torch.FloatTensor (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed

    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)

    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself

    Return
    ------
    flat_grads : torch.FloatTensor
        a flattened view of the gradients of functional_output w.r.t. inputs
    '''

    if create_graph == True:
        retain_graph = True

    grads = torch.autograd.grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = flatten(grads)

    return flat_grads


def get_flat_params(parameterized_fun):
    '''
    Get a flattened view of the parameters of a function approximator

    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator for which the parameters are to be returned

    Returns
    -------
    flat_params : torch.FloatTensor
        a flattened view of the parameters of parameterized_fun
    '''
    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.view(-1) for param in parameters])

    return flat_params

def detach_dist(dist):
    '''
    Return a copy of dist with the distribution parameters detached from the
    computational graph

    Parameters
    ----------
    dist: torch.distributions.distribution.Distribution
        the distribution object for which the detached copy is to be returned

    Returns
    -------
    detached_dist
        the detached distribution
    '''

    if type(dist) is torch.distributions.categorical.Categorical:
        detached_dist = torch.distributions.categorical.Categorical(logits=dist.logits.detach())
    elif type(dist) is torch.distributions.Independent:
        detached_dist = torch.distributions.normal.Normal(loc=dist.mean.detach(), scale=dist.stddev.detach())
        detached_dist = torch.distributions.Independent(detached_dist, 1)

    return detached_dist

class RunningStat:
    '''
    Keeps track of a running estimate of the mean and standard deviation of
    a distribution based on the observations seen so far

    Attributes
    ----------
    _M : torch.float
        estimate of the mean of the observations seen so far

    _S : torch.float
        estimate of the sum of the squared deviations from the mean of the
        observations seen so far

    n : int
        the number of observations seen so far

    Methods
    -------
    update(x)
        update the running estimates of the mean and standard deviation

    mean()
        return the estimated mean

    var()
        return the estimated variance

    std()
        return the estimated standard deviation
    '''

    def __init__(self):
        self._M = None
        self._S = None
        self.n = 0

    def update(self, x):
        self.n += 1

        if self.n == 1:
            self._M = x.clone()
            self._S = torch.zeros_like(x)
        else:
            old_M = self._M.clone()
            self._M = old_M + (x - old_M) / self.n
            self._S = self._S + (x - old_M) * (x - self._M)

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self.n > 1:
            var = self._S / (self.n - 1)
        else:
            var = torch.pow(self.mean, 2)

        return var

    @property
    def std(self):
        return torch.sqrt(self.var)


class Transform:
    '''
    Composes several transformation and applies them sequentially

    Attributes
    ----------
    filters : list
        a list of callables

    Methods
    -------
    __call__(x)
        sequentially apply the callables in filters to the input and return the
        result
    '''

    def __init__(self, *filters):
        '''
        Parameters
        ----------
        filters : variatic argument list
            the sequence of transforms to be applied to the input of
            __call__
        '''

        self.filters = list(filters)

    def __call__(self, x):
        for f in self.filters:
            x = f(x)

        return x


class ZFilter:
    '''
    A z-scoring filter

    Attributes
    ----------
    running_stat : RunningStat
        an object that keeps track of an estimate of the mean and standard
        deviation of the observations seen so far

    Methods
    -------
    __call__(x)
        Update running_stat with x and return the result of z-scoring x
    '''

    def __init__(self):
        self.running_stat = RunningStat()

    def __call__(self, x):
        self.running_stat.update(x)
        x = (x - self.running_stat.mean) / (self.running_stat.std + 1e-8)

        return x


class Bound:
    '''
    Implements a bounding function

    Attributes
    ----------
    low : int
        the lower bound

    high : int 
        the upper bound

    Methods
    -------
    __call__(x)
        applies the specified bounds to x and returns the result
    '''

    def __init__(self, low, high):
        '''
        Parameters
        ----------
        low : int
            the lower bound

        high : int
            the upper bound
        '''
        
        self.low = low
        self.high = high

    def __call__(self, x):
        x = torch.clamp(x, self.low, self.high)

        return x
