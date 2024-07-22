import matplotlib.pyplot as plt
import re
def draw_loss_curve(log_path, key_strs=['loss']):
    with open(log_path, 'r') as f:
        a = f.readlines()
        for key_str in key_strs:
            a = filter(lambda x:'loss' in x or 'lr' in x, a)
            a = list(map(lambda x:x[:-1], a))
            pattern_loss = re.compile(f'{key_str}: *(\d+\.\d+)')
            loss = list(map(lambda x:float(pattern_loss.findall(x)[0]), a))
            key_str = key_str.title()
            plt.plot(loss)
            plt.xlabel('Steps')
            plt.ylabel(f'{key_str}')
            plt.title(f'Training {key_str} Over Steps')
            plt.grid(True)
            plt.show()
