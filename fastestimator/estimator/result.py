

class Result:
    def __init__(self):
        # Users will have access to self.network will be the Network instance user defined
        pass

    def begin(self, mode):
        """this function runs at the beginning of the mode
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
        """
        pass

    def on_epoch_begin(self, mode, logs):
        """this function runs at the beginning of each epoch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
        """
        pass

    def on_batch_begin(self, mode, logs):
        """this function runs at the beginning of every batch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
                "step": current global step index starting from 0 (or batch index)
                "size": current batch size
        """
        pass

    def on_batch_end(self, mode, logs):
        """this function runs at the end of every batch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
                "step": current global step index starting from 0 (or batch index)
                "size": current batch size
                "prediction": the batch predictions in dictionary format
                "loss": the batch loss (only available when mode is "train" or "eval")
        """
        pass

    def on_epoch_end(self, mode, logs):
        """this function runs at the end of every epoch
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
            logs (dict): dictionary with following key:
                "epoch": current epoch index starting from 0
        """
        pass

    def end(self, mode):
        """this function runs at the end of the mode
        
        Args:
            mode (str): signature during different phases, can be "train", "eval" or "test"
        """
        pass