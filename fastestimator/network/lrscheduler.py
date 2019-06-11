import numpy as np
import math

class CyclicScheduler:
    """
    A class representing cyclic learning rate scheduler

    Args:
        num_cycle: The number of cycles to be used by the learning rate scheduler
        cycle_multiplier: The length of each next cycle's multiplier
        decrease_method: The decay method to be used with cyclic learning rate scheduler
    """
    def __init__(self, num_cycle=1, cycle_multiplier=2, decrease_method="cosine"):
        self.mode = "global_steps"
        self.num_cycle = num_cycle
        self.cycle_multiplier = cycle_multiplier
        self.decrease_method = decrease_method
        self.epochs = None
        self.steps_per_epoch = None
        self.offset = 0
        self.decay_fn_map = {"cosine": self.lr_cosine_decay,
                             "linear": self.lr_linear_decay}
        if self.cycle_multiplier < 1:
            raise ValueError("The cycle multiplier should at least be 1")

    def lr_linear_decay(self, global_steps, lr_ratio_start, lr_ratio_end, step_start, step_end):
        slope = (lr_ratio_start - lr_ratio_end)/(step_start - step_end)
        intercept = lr_ratio_start - slope * step_start
        lr_ratio = slope * global_steps + intercept
        return np.float32(lr_ratio)

    def lr_cosine_decay(self, global_steps, lr_ratio_start, lr_ratio_end, step_start, step_end):
        current_cycle = (global_steps - step_start)/(step_end - step_start)
        lr_ratio = (lr_ratio_start - lr_ratio_end)/2 * math.cos(current_cycle*math.pi) + (lr_ratio_start + lr_ratio_end)/2
        return np.float32(lr_ratio)

    def lr_schedule_fn(self, global_steps):
        """
        The actual function that computes the learning rate decay ratio using cyclic learning rate.
        
        Args:
            global_steps: Current global step

        Returns:
            Learning rate ratio
        """
        if self.cycle_multiplier == 1:
            total_unit_cycles = self.num_cycle
        else:
            total_unit_cycles = (self.cycle_multiplier**self.num_cycle - 1)/(self.cycle_multiplier - 1)
        total_steps = self.epochs * self.steps_per_epoch - self.offset
        unit_cycle_length = total_steps // total_unit_cycles
        if global_steps < self.offset:
            lr_ratio = 1.0
        else:
            step_start = self.offset
            for i in range(self.num_cycle):
                current_cycle_length = unit_cycle_length*self.cycle_multiplier**i
                if (global_steps - step_start) < current_cycle_length:
                    step_end = step_start + current_cycle_length
                    break
                else:
                    if i == (self.num_cycle-1):
                        step_end = total_steps + self.offset
                    else:
                        step_start = step_start + current_cycle_length
            lr_ratio = self.decay_fn_map[self.decrease_method](global_steps, 1.0, 1e-6, step_start, step_end)
        return np.float32(lr_ratio)
