import os
import pymmcore_plus
from .abstract_microscope import AbstractMicroscope


class MMDemo(AbstractMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    MICROMANAGER_CONFIG = "C:\\Program Files\\Micro-Manager-2.0\\MMConfig_demo.cfg"
    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False

    def __init__(self):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")
        self.mmc.setChannelGroup(channelGroup=self.CHANNEL_GROUP)

    def run_experiment(self):
        """Run the experiment."""
        pymmcore_plus.configure_logging(stderr_level="WARNING")

    def post_experiment(self):
        """Post-process the experiment."""
        pass
