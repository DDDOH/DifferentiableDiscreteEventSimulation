import matplotlib.pyplot as plt
class Logger:
    def __init__(self):
        self.logs = {}
    
    def add(self, name, step, value):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append((step, value))
        
        
    def visualize(self, name, path, ylabel, xlabel, log_scale=False):
        plt.figure()
        plt.plot(*zip(*self.logs[name]))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if log_scale:
            plt.yscale('log')
        plt.savefig(path)
        plt.close()
        