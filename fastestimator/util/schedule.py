class Scheduler:
    def __init__(self, epoch_dict):
        self.epoch_dict = epoch_dict
        self._verify_inputs()

    def _verify_inputs(self):
        assert isinstance(self.epoch_dict, dict), "must provide dictionary as epoch_dict"
        self.keys = list(self.epoch_dict.keys())
        self.keys.sort()
        for key in self.keys:
            assert isinstance(key, int), "found non-integer key: {}".format(key)
    
    def _get_current_value(self, epoch):
        if epoch in self.keys:
            value = self.epoch_dict[epoch]
        else:
            last_key = self._get_last_key(epoch)
            if last_key is None:
                value = None
            else:
                value = self.epoch_dict[last_key]
        return value

    def _get_last_key(self, epoch):
        if epoch < min(self.keys):
            last_key = None
        else:
            for key in self.keys:
                if key < epoch:
                    last_key = key
                else:
                    break
        return last_key
