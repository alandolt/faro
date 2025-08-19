"""Image processing pipeline package.

This package provides modular image processing pipeline implementations
for different storage backends and processing scenarios.

Available pipelines:
- BaseImageProcessingPipeline: Abstract base class defining the pipeline interface
- ImageProcessingPipeline: Standard TIFF-based pipeline for live image processing
- ImageProcessingPipelineOmeZarr: OME-Zarr based pipeline for efficient multidimensional data storage
- ImageProcessingPipeline_postExperiment: Specialized pipeline for reprocessing existing data
"""
