import numpy as np
import math

class EMA(object):
    def __init__(self, nepochs, iter_per_epochs, start_value=0.996, end_value=1.0):
        
        self.decay = cosine_decay(
                    start_value=start_value, 
                    end_value=end_value,
                    epochs=nepochs,
                    iter_per_epochs=iter_per_epochs
                )

    def __call__(self, student_model, teacher_model, step):
        
        
        for std_param, t_param in zip(
            student_model.parameters(), teacher_model.parameters()
        ):
            t_param.data = self.decay[step] * t_param.data + (1 - self.decay[step]) * std_param.data


def cosine_decay(start_value, end_value, epochs, iter_per_epochs, start_warmup_epoch=0, warmup_epochs=0):
    '''
        value(t) = end_value + (start_value - end_value) * (1 + cos(π * t / T)) / 2
    '''
    warmup_iters = warmup_epochs * iter_per_epochs
    if warmup_epochs > 0:
        linear_decay = np.linspace(start=start_warmup_epoch, stop=start_value, num=warmup_iters)
    
    total_iters = epochs * iter_per_epochs
    decay_iters = np.arange(stop=(total_iters - warmup_iters)) 

    decay_factor = (1 + np.cos(np.pi * decay_iters / total_iters)) / 2

    cosine_decay = end_value + (start_value - end_value) * decay_factor

    if warmup_epochs > 0:
        return np.concatenate([linear_decay, cosine_decay])
    return cosine_decay


if __name__ == "__main__":
    
    # for step in range(1001):
    value = cosine_decay(1e-6, (0.0005 * 32/256), 30, 3, warmup_epochs=10)
    print(f"{value}")