# consider single server queue with poisson arrival and arbitrary service time distribution
# use Gaussian Mixture distribution for service time distribution, or maybe simple discrete distribution
# assume we can observe customer arrival time and wait time
from diff_des.diff_des import PoissonArrival, ExponentialService, DiscreteService, SingleServerQueue, BatchSequence
import torch
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import os
from datetime import datetime as datatime
from utils import Logger

if __name__ == '__main__':
    import shutil
    RESULT_DIR = os.path.join('/Users/shuffleofficial/Offline_Documents/24-11-26 Differentiable DES/tmp', datatime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(RESULT_DIR)
    code_dir = os.path.join(RESULT_DIR, 'code')
    os.mkdir(code_dir)
    result_dir = os.path.join(RESULT_DIR, 'result')
    os.mkdir(result_dir)
    # copy all files and folders into result directory, except for tmp.nosync folder
    for filename in os.listdir('.'):
        if filename not in ['tmp.nosync', '.idea', '.vscode', 'checkpoint.nosync']:
            src_path = os.path.join('.', filename)
            dest_path = os.path.join(code_dir, filename)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy(src_path, dest_path)
                


    ############## Discrete Service Time ################
    # plot mean wait time as a function of service rate, and the gradient of service rate w.r.t. mean wait time
    # assume service time is discrete with two values, 2 and 3 with probability p and 1-p, p is unknown
    if True:
        
        from scipy.stats import expon
        import ot
        
        def visualize_result(service_dist: DiscreteService, simulate_wait_time, true_wait_time, grad, filename):
            type = 'cdf'
            if type == 'pdf':
                raise NotImplemented
                # Define the x-values for the exponential PDF
                x = np.linspace(0, 10, 100)  # Generate values up to the max service time
                pdf = expon.pdf(x, scale=1)  # Exponential distribution with rate parameter λ=1
                plt.figure()
                plt.scatter(vals, probs, alpha=0.6, label='Normalized Probabilities', marker='o')  # Plot the discrete distribution
                plt.plot(x, pdf, 'r-', lw=2, label='Exponential PDF (λ=1)')  # Plot the exponential PDF
                plt.xlabel("Service Time")
                plt.ylabel("Probability")
                plt.legend()
                plt.savefig(filename)
                plt.close()
            elif type == 'cdf':
                
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                service_dist.plot_cdf()
                # plot gradient that scales to [0, 1]
                scaled_grad = grad / grad.abs().max()
                plt.bar(service_dist.vals, scaled_grad.detach().numpy(), alpha=0.6, label='Gradient')
                
                x = np.linspace(0, 10, 100)
                cdf = expon.cdf(x, scale=1)
                plt.plot(x, cdf, 'r-', lw=2, label='Exponential CDF (λ=1)')
                plt.xlabel("Service Time")
                plt.ylabel("CDF")
                plt.legend()
                plt.subplot(122)
                # plot ecdf of simulate_wait_time and true_wait_time
                concat_simulate_wait_time = torch.cat(simulate_wait_time)
                plt.plot(np.sort(concat_simulate_wait_time), np.linspace(0, 1, len(concat_simulate_wait_time), endpoint=False), label='Simulated Wait Time')
                
                concat_true_wait_time = torch.cat(true_wait_time)
                plt.plot(np.sort(concat_true_wait_time), np.linspace(0, 1, len(concat_true_wait_time), endpoint=False), label='True Wait Time')
                
                plt.xlabel("Wait Time")
                plt.ylabel("ECDF")
                plt.legend()
                plt.savefig(filename)
                plt.close()
                
                
        def loss_fn(target_mean, target_std, result) -> torch.Tensor:
            mean = torch.stack([_.mean() for _ in result.batch_wait_times])
            std = torch.stack([_.std() for _ in result.batch_wait_times])
            return (mean - target_mean).pow(2) + (std - target_std).pow(2)
            
            
            # wait_times = [_.wait_times for _ in result]
            # # only consider the mean wait time
            # loss = (wait_times.mean() - true_wait_times.mean())**2 + (wait_times.std() - true_wait_times.std())**2
            # # OT
            # # M = ot.dist(x1=wait_times.unsqueeze(1), x2=true_wait_times.unsqueeze(1))
            # # loss = ot.emd2(torch.ones_like(wait_times) / len(wait_times),
            # #             torch.ones_like(true_wait_times) / len(true_wait_times), M)
            
            
        
        # Ground truth data
        T = 500
        np.random.seed(0)
        torch.random.manual_seed(0)
        arrival = PoissonArrival(0.95)
        service = ExponentialService(1.0)
        ssq = SingleServerQueue(arrival, service, batch_size=100, enable_multiprocessing=False)
        true_result = ssq.simulate(T)
        
        BatchSequence.plot_batch(true_result.batch_arrival_time, true_result.batch_wait_times, color='C0', alpha=0.3)
        target_mean = torch.stack([_.mean() for _ in true_result.batch_wait_times]).mean()
        target_std = torch.stack([_.std() for _ in true_result.batch_wait_times]).mean()
        plt.xlabel("Arrival Time")
        plt.ylabel("Wait Time")
        plt.savefig(os.path.join(result_dir,"wait_time.png"))
        plt.close()
        
        
        # now we want to estimate the service rate mu using the observed arrival times and wait times
        arrival = PoissonArrival(0.95)
        n_mass = 20
        vals = np.linspace(0, 10, n_mass)
        prob_init_val = np.ones(n_mass) / n_mass
        # prob_init_val[:n_mass // 2] = 0
        # prob_init_val[n_mass // 2:] = 0
        
        service = DiscreteService(vals, torch.tensor(prob_init_val, requires_grad=True))
        
        n_sim = 50000
        batch_size = 3 # large batch size converges slower, reason unknown
        
        ssq = SingleServerQueue(arrival, service, batch_size=batch_size)

        optimizer = torch.optim.Adam([service.raw_probs], lr=5e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=n_sim)
        logger = Logger()
        
        for i in progressbar.progressbar(range(n_sim//batch_size)):
            result = ssq.simulate(T)
            
            loss = loss_fn(target_mean, target_std, result)
            
            log_prob_batch = torch.stack([service.log_prob(_).sum() for _ in result.batch_service_times])
            _ = (loss * log_prob_batch).mean()
            optimizer.zero_grad()
            _.backward()
            
            logger.add(name='loss', step=i * batch_size, value=loss.mean().item())
            
            # print before the gradient step
            if i % (1000 //batch_size) == 0:
                # print learning rate
                print(f"lr={scheduler.get_last_lr()}")
                print(f"\nloss={loss}, probs={service.raw_probs}")
                
                visualize_result(service,
                                result.batch_wait_times,
                                true_result.batch_wait_times,
                                service.raw_probs.grad,
                                os.path.join(result_dir, f"service_time_{i * batch_size}.png"))

                logger.visualize('loss', os.path.join(result_dir, "loss.png"), ylabel="Loss", xlabel="Iteration", log_scale=True)
            
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                service.raw_probs.data.clamp_(min=0)
                # might use gumbel-max trick? https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/