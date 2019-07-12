import tensorflow as tf

class Operation:
    def __init__(self, key_in, link, key_out):
        """Operation class that represent the forward operation of network
        
        Args:
            key_in (str, list): the key of input data from batch data dictionary
            link (block, list): operation or list of operations performed on input data
            key_out (str): the key to store operation result
        """
        self.key_in = key_in
        self.key_out = key_out
        self.link = link
        self._check_block()
    
    def _check_block(self):
        if not isinstance(self.link, list):
            self.link = [self.link]
        for block in self.link:
            assert hasattr(block, "__call__"), "object: {} is not callable".format(block)

    def forward(self, batch, mode, epoch):
        data = batch[self.key_in]
        for block in self.link:
            if isinstance(block, tf.keras.Model) or block.mode in [mode, "both"]:
                data = block(data)
        batch[self.key_out] = data
        return batch