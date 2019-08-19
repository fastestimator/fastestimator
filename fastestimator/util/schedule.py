import tensorflow as tf

from fastestimator.util.op import TensorOp


class Scheduler:
    def __init__(self, epoch_dict):
        self.epoch_dict = epoch_dict
        self._verify_inputs()

    def _verify_inputs(self):
        assert isinstance(self.epoch_dict, dict), "must provide dictionary as epoch_dict"
        self.keys = list(self.epoch_dict.keys())
        self.keys.sort()
        sample_content = self._get_sample_content()
        for key in self.keys:
            assert isinstance(key, int), "found non-integer key: {}".format(key)
            if isinstance(sample_content, TensorOp) and self.epoch_dict[key]:
                assert self.mode == self.epoch_dict[key].mode, "schedule contents must have same mode"

    def get_current_value(self, epoch):
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

    def _get_sample_content(self):
        for key in self.keys:
            if self.epoch_dict[key] is not None:
                sample_content = self.epoch_dict[key]
                break
        if isinstance(sample_content, TensorOp):
            self.mode = sample_content.mode
        return sample_content
