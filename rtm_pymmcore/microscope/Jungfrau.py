import pymmcore_plus
from rtm_pymmcore.microscope.abstract_microscope import AbstractMicroscope


class Jungfrau(AbstractMicroscope):
    MICROMANAGER_PATH = "C:\\Program Files\\Micro-Manager-2.0"
    MICROMANAGER_CONFIG = "E:\\pertzlab_mic_configs\\micromanager\\\Jungfrau\\TiFluoroJungfrau_w_TTL_DIGITALIO.cfg"
    USE_AUTOFOCUS_EVENT = False
    USE_ONLY_PFS = True

    def __init__(self):
        super().__init__()
        pymmcore_plus.use_micromanager(self.MICROMANAGER_PATH)
        self.mmc = pymmcore_plus.CMMCorePlus()

    def init_scope(self):
        """Initialize the microscope."""
        self.mmc.loadSystemConfiguration(self.MICROMANAGER_CONFIG)
        self.mmc.setConfig(groupName="System", configName="Startup")

    def run_experiment(self):
        """Run the experiment."""
        pymmcore_plus.configure_logging(stderr_level="WARNING")

    def post_experiment(self):
        """Post-process the experiment."""
