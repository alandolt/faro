"""Base class for image processing pipelines."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from useq import MDAEvent

import rtm_pymmcore.segmentation.base_segmentation as base_segmentation
import rtm_pymmcore.stimulation.base_stimulation as base_stimulation
import rtm_pymmcore.tracking.abstract_tracker as abstract_tracker
import rtm_pymmcore.feature_extraction.abstract_fe as abstract_fe


class BaseImageProcessingPipeline(ABC):
    """
    Abstract base class for image processing pipelines.

    """

    @abstractmethod
    def store_img(
        self, img: np.ndarray, metadata: Dict[str, Any], path: str, folder: str
    ) -> None:
        """
        Store an image to the specified location.

        Args:
            img: Image array to store
            metadata: Metadata associated with the image
            path: Base path for storage
            folder: Specific folder within the path
        """
        pass

    @abstractmethod
    def run(self, img: np.ndarray, event: MDAEvent) -> Dict[str, Any]:
        """
        Run the image processing pipeline on the input image.

        Args:
            img: Input image to process
            event: MDAEvent containing metadata

        Returns:
            Dictionary containing pipeline results
        """
        pass
