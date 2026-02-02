class Agent:
    """
    Base class for all Agents. Specific implementations should inherit
    from this class and override this method.
    """

    def __init__(self, pipeline, microscope):
        pass

    def add_fovs(self, fovs: list):
        raise NotImplementedError("Subclasses should implement this!")

    def run(self):
        raise NotImplementedError("Subclasses should implement this!")

    def stop(self):
        raise NotImplementedError("Subclasses should implement this!")
