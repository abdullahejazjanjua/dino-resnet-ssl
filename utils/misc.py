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
    def __init__(self):

        self.decay = self._cosine_decay()

    def update(self, student_model, teacher_model):
        for std_param, t_param in zip(
            student_model.parameters(), teacher_model.parameters()
        ):
            t_param.data = self.decay * t_param.data + (1 - self.decay) * std_param.data

    def _cosine_decay():
        pass
