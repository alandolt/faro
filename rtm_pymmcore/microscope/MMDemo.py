import pymmcore_plus
from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope
from rtm_pymmcore.controller import ControllerSimulated, Analyzer


class MMDemo(AbstractMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    MICROMANAGER_CONFIG = "C:\\Program Files\\Micro-Manager-2.0\\MMConfig_demo.cfg"
    CHANNEL_GROUP = "Channel"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = False

    def __init__(self, old_data_project_path: str):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()
        self.old_data_project_path = old_data_project_path
        self.init_scope()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")
        self.mmc.setChannelGroup(channelGroup=self.CHANNEL_GROUP)

    def run_experiment(self, df_acquire):
        """Run the experiment."""
        self.analyzer = Analyzer(self.pipeline)
        self.controller = ControllerSimulated(
            self.analyzer,
            self.mmc,
            self.queue,
            self.USE_AUTOFOCUS_EVENT,
            project_path=self.old_data_project_path,
        )
        pymmcore_plus.configure_logging(stderr_level="WARNING")

    def post_experiment(self):
        """Post-process the experiment."""
        pass
