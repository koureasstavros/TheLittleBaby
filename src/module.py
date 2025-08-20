#########################
# Base Module Definition
# Author: Koureas Stavros
#########################

class Module:
    """
    Base class for all neural network modules.
    Manages parameters and training/evaluation mode.
    """
    def __init__(self):
        self.setting = True # Default mode for modules
        self._parameters = [] # List to store parameters

    def parameters(self):
        """
        Returns a list of all parameters in the module.
        This method should be overridden by subclasses.
        """
        return self._parameters

    def set(self, mode=True):
        """
        Sets the module and all its sub-modules to training mode.
        If mode is False, sets to evaluation mode.
        """
        self.setting = mode
        # Recursively set training mode for any sub-modules
        for param in self._parameters: # Iterate through parameters, which might be other Modules
            if isinstance(param, Module):
                param.set(mode)
            # For actual numpy arrays (weights/biases), the `training` flag is used by Dropout
            # and other layers that behave differently in train/eval.

    def eval(self):
        """Sets the module to evaluation mode."""
        self.set(False)