import matplotlib.pyplot as plt
import re
def draw_loss_curve(log_path):
    with open(log_path, 'r') as f:
        a = f.readlines()
        a = filter(lambda x:'loss' in x or 'lr' in x, a)
        a = list(map(lambda x:x[:-1], a))
        pattern_loss = re.compile(r'loss:(\d+\.\d+)')
        loss = list(map(lambda x:float(pattern_loss.findall(x)[0]), a))
        plt.plot(loss)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Steps')
        plt.grid(True)
        plt.show()
