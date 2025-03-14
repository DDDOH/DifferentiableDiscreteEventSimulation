from diff_des import PoissonArrival, ExponentialService, DiscreteService, SingleServerQueue, BatchSequence
import torch
import matplotlib.pyplot as plt
import progressbar
import numpy as np
############## Exponential Service Time ################


if __name__ == "__main__":
    # generate ground truth data 
    T = 100
    arrival = PoissonArrival(8)
    real_service = ExponentialService(5.0)

    ssq = SingleServerQueue(arrival, real_service, batch_size=100, enable_multiprocessing=False)
    result = ssq.simulate(T)
    target_mean = result.mean('wait_times').mean()
    del ssq

    # plot mean wait time as a function of service rate, and the gradient of service rate w.r.t. mean wait time
    mu_ls = torch.linspace(0.1, 10.0, 100)
    mean_wait_times = []
    grad_ls = []
    for mu_val in mu_ls:
        mu = torch.tensor(mu_val, requires_grad=True)
        service = ExponentialService(mu)
        ssq = SingleServerQueue(arrival, service, batch_size=1, enable_multiprocessing=False)
        result = ssq.simulate(T)
        
        mean_wait_times.append(result.mean('wait_times').detach().item())
        grad = torch.autograd.grad(result.mean('wait_times'), service.mu)[0]
        grad_ls.append(grad)
        
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(mu_ls, torch.as_tensor(mean_wait_times))
    plt.xlabel("Service Rate")
    plt.ylabel("Mean Wait Time")
    plt.subplot(2, 1, 2)
    plt.plot(mu_ls, grad_ls)
    plt.xlabel("Service Rate")
    plt.ylabel("Gradient of Mean Wait Time")
    plt.tight_layout()
    plt.savefig("mean_wait_time.png")


    # now we want to estimate the service rate mu using the observed arrival times and wait times
    # we can use gradient descent to minimize the loss function
    T = 100
    arrival = PoissonArrival(8)
    init_service_rate = torch.tensor(2.0, requires_grad=True)
    service = ExponentialService(init_service_rate)
    
    calibrated_service_rate_ls = []
    loss_ls = []
    
    ssq = SingleServerQueue(arrival, service, batch_size=1, enable_multiprocessing=False)
    
    end_lr = 1e-5
    start_lr = 5e-2
    optimizer = torch.optim.Adam([service.mu], lr=start_lr)
    n_iter = 10000
    
    # gamma = (end_lr / start_lr)**(1/n_iter)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=end_lr/start_lr, total_iters=n_iter)
    for i in progressbar.progressbar(range(n_iter)):
        calibrated_service_rate_ls.append(service.mu.item())
        result = ssq.simulate(T)
        wait_times = result.batch_wait_times[0]
        loss = (wait_times.mean() - target_mean)**2
        loss_ls.append(loss.item())
        # print(f"loss={loss}, mu={service.mu}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        # service.mu.data -= 0.0005 * service.mu.grad
        # service.mu.grad.zero_()
        
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogy(loss_ls)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.subplot(2, 1, 2)
    plt.plot(calibrated_service_rate_ls, label="Calibrated Service Rate")
    plt.xlabel("Iteration")
    plt.ylabel("Service Rate")
    plt.hlines(real_service.mu, 0, len(calibrated_service_rate_ls), color='r', label="True Service Rate")
    plt.savefig("calibrated_service_rate.png")

