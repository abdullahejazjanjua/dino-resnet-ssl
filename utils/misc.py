import math
from collections import defaultdict


class MetricLogger(object):
    def __init__(self):
        self.log = defaultdict(list)

    def __call__(self, key, value):
        self.log.update({key: value})

    def log_info(self):
        for k, v in self.log.items():
            print(f"{k}: {v}")


class EMA:
    def __init__(self, total_steps, current_step):

        self.decay = EMA._cosine_decay(start_value=0.996, end_value=1, total_steps=total_steps, current_step=current_step)

    def __call__(self, student_model, teacher_model):
        for std_param, t_param in zip(
            student_model.parameters(), teacher_model.parameters()
        ):
            t_param.data = self.decay * t_param.data + (1 - self.decay) * std_param.data
    
    @staticmethod
    def _cosine_decay(start_value, end_value, total_steps, current_step):
        '''
        value(t) = end_value + (start_value - end_value) * (1 + cos(π * t / T)) / 2
        '''
        progress = current_step / total_steps
        decay_factor = (1 + math.cos(math.pi * progress)) / 2
        return end_value + (start_value - end_value) * decay_factor


if __name__ == "__main__":
    
    for step in range(51):
        value = EMA._cosine_decay(100, 10, 50, step)
        print(f"Step {step}: {value:.2f}")