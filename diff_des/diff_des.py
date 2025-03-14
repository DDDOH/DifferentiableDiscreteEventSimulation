
import numpy as np
import torch
import torch.distributions as dist
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


class BatchSequence:
    @staticmethod
    def plot_batch(seqs_1, seqs_2, color=None, alpha=0.5):
        n_seq = len(seqs_1)
        for i in range(n_seq):
            plt.plot(seqs_1[i], seqs_2[i], c=color, alpha=alpha)


class BatchSimulationResult:
    def __init__(self, batch_arrival_time, batch_service_times, batch_wait_times, batch_exit_times):
        self.batch_arrival_time = batch_arrival_time
        self.batch_service_times = batch_service_times
        self.batch_wait_times = batch_wait_times
        self.batch_exit_times = batch_exit_times
        self.batch_size = len(batch_arrival_time)

        # assert each batch has the same number of sequences
        assert len(batch_arrival_time) == len(batch_service_times) == len(batch_wait_times) == len(batch_exit_times)

        for i in range(self.batch_size):
            # for each sequence, assert the number of arrivals is the same
            assert len(batch_arrival_time[i]) == len(batch_service_times[i]) == len(batch_wait_times[i]) == len(batch_exit_times[i])
            torch.allclose(batch_arrival_time[i] + batch_service_times[i] + batch_wait_times[i], batch_exit_times[i])
        
    def __repr__(self):
        """
        Print out the simulation result with basic statistics.
        """
        n = len(self.arrival_times)
        avg_wait = torch.mean(self.wait_times)
        avg_service = torch.mean(self.service_times)
        avg_exit = torch.mean(self.exit_times)
        avg_interarrival = torch.mean(self.arrival_times[1:] - self.arrival_times[:-1])
        
        return f"SimulationResult(n={n}, avg_wait={avg_wait}, avg_service={avg_service}, avg_exit={avg_exit}, avg_interarrival={avg_interarrival})"
    
    # implement + operator
    def __add__(self, other):
        return BatchSimulationResult(
            self.batch_arrival_time + other.batch_arrival_time,
            self.batch_service_times + other.batch_service_times,
            self.batch_wait_times + other.batch_wait_times,
            self.batch_exit_times + other.batch_exit_times,
        )
        
    def mean(self, label: str):
        """
        Compute the mean of the given label across all sequences.
        
        Args:
            label: str, the label to compute the mean. It can be 'arrival_times', 'service_times', 'wait_times', 'exit_times'.
        """
        return torch.stack([getattr(self, 'batch_'+label)[i].mean() for i in range(self.batch_size)])


class PoissonArrival:
    def __init__(self, lam):
        """
        Possion arrival process with rate lam.
        Interarrival time is exponentially distributed with rate lam.
        """
        self.lam = lam
        
    def sample_up_to(self, T):
        """
        Sample arrival times up to time T.
        """
        arrival_times = []
        t = 0
        while t < T:
            t += dist.Exponential(self.lam).sample()
            arrival_times.append(t.clone())
        return torch.as_tensor(arrival_times)
        
class ExponentialService:
    def __init__(self, mu):
        """
        Exponential service time with rate mu.
        """
        self.mu = mu
        
    def sample(self, n):
        """
        Sample n service times.
        """
        return dist.Exponential(self.mu).rsample((n,))
    
class DiscreteService:
    def __init__(self, vals, probs):
        """
        Discrete service time sampling that supports REINFORCE gradient estimation.
        
        raw_probs is passed into Sigmoid to ensure that the probabilities are in (0, 1).
        """
        self.vals = torch.tensor(vals, dtype=torch.float32)
        self.raw_probs = probs
        self.vals_2_indices = {val.item(): i for i, val in enumerate(self.vals)}
        # use linear scale
        self.scale = lambda x: x
        # torch.nn.Sigmoid() # lambda x: x
        
    def sample(self, n):
        """
        Sample service times using the custom autograd Function.
        """
        return self.vals[dist.Categorical(self.scale(self.raw_probs)).sample((n,))]
    
    def log_prob(self, samples):
        """
        Compute the log probability of the samples.
        """
        indices = [self.vals_2_indices[sample.item()] for sample in samples]
        return dist.Categorical(self.scale(self.raw_probs)).log_prob(torch.as_tensor(indices))
    
    def plot_cdf(self):
        """
        Plot the CDF of the service time distribution.
        """
        valid_probs = self.scale(self.raw_probs)
        normed_probs = (valid_probs / valid_probs.sum()).detach().numpy()
        x_ls = np.linspace(min(self.vals), max(self.vals), 100)
        ecdf = lambda x: np.sum(normed_probs[self.vals <= x])
        y_ls = [ecdf(x) for x in x_ls]
        plt.plot(x_ls, y_ls)
        
                
def simulate_batch(batch):
    """Simulate a single batch for the queue."""
    batch_arrival_times, batch_service_times, local_rank = batch
    n_arrival = len(batch_arrival_times)
    relu = torch.nn.ReLU()
    wait_times = torch.zeros(n_arrival)
    exit_times = torch.zeros(n_arrival)

    # Simulating server queue for the batch
    idle_time = 0  # Assume idle_time starts fresh for each batch
    for i in range(n_arrival):
        wait_times[i] = relu(idle_time - batch_arrival_times[i])
        idle_time = batch_arrival_times[i] + wait_times[i] + batch_service_times[i]
        exit_times[i] = idle_time

        assert torch.isclose(batch_arrival_times[i] + batch_service_times[i] + wait_times[i], exit_times[i])
    return wait_times, exit_times
        
        
class SingleServerQueue:
    def __init__(self, arrival, service, batch_size=1, enable_multiprocessing=True):
        self.arrival = arrival
        self.service = service
        self.batch_size = batch_size
        self.pool = None  # Persistent process pool
        self.enable_multiprocessing = enable_multiprocessing

    def start_pool(self):
        """Initialize the multiprocessing pool."""
        if self.pool is None:
            self.pool = mp.Pool(self.batch_size)

    def stop_pool(self):
        """Close and terminate the pool."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def simulate(self, T):
        if self.enable_multiprocessing:
            """Simulate the queue with multiprocessing."""
            self.start_pool()  # Ensure the pool is running
                        
            # Create batches for multiple simulation replications
            batches = []
            for i in range(self.batch_size):
                arrival_times = self.arrival.sample_up_to(T)
                n_arrivals = len(arrival_times)
                service_times = self.service.sample(n_arrivals)
                batches.append((arrival_times, service_times, i))

            # Use the persistent pool to process batches
            results = self.pool.map(simulate_batch, batches)

            # Combining results from all batches
            wait_times_list, exit_times_list = zip(*results)

            return BatchSimulationResult(
                batch_arrival_time=[_[0] for _ in batches],
                batch_service_times=[_[1] for _ in batches],
                batch_wait_times=wait_times_list,
                batch_exit_times=exit_times_list,
            )
        else:
            """Simulate the queue without multiprocessing."""
            result = BatchSimulationResult([], [], [], [])
            for i in range(self.batch_size):
                arrival_times = self.arrival.sample_up_to(T)
                n_arrivals = len(arrival_times)
                service_times = self.service.sample(n_arrivals)
                wait_times, exit_times = simulate_batch((arrival_times, service_times, 0))
                result += BatchSimulationResult(
                    batch_arrival_time=[arrival_times],
                    batch_service_times=[service_times],
                    batch_wait_times=[wait_times],
                    batch_exit_times=[exit_times],
                )
            return result
            
        
    def __del__(self):
        self.stop_pool()