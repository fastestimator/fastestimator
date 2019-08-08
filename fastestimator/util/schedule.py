
class Scheduler:
    def __init__(self, mode="epoch", schedule_fn=None, schedule_dict=None):
        self.mode = mode
        self.schedule_fn = schedule_fn
        self.schedule_dict = schedule_dict
        assert self.mode in ["epoch", "step"], "mode should either be 'epoch' or 'step'"
        assert bool(self.schedule_fn) != bool(self.schedule_dict), "should provide either 'shedule_fn' or 'schedule_dict'"

    def get_current_value(self, state):
        x = state[self.mode].numpy()
        if self.schedule_dict:
            value = self.schedule_dict[x]
        else:
            value = self.schedule_fn(x)
        return value
    
    def change_value(self, state):
        x = state[self.mode].numpy()
        change = True
        if self.schedule_dict and x not in list(self.schedule_dict.keys()):
            change = False
        return change