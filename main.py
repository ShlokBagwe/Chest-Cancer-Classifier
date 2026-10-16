import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # flexible async allocator

import tensorflow as tf
# Must configure memory growth BEFORE any pipeline stage imports initialize TF GPU
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from ChestCancer.pipeline.stage_03_baseline_model import PrepareBaseModelPipeline
from ChestCancer.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from ChestCancer import logger
from ChestCancer.pipeline.stage_02_data_preparation import DataPreparationPipeline
from ChestCancer.pipeline.stage_04_model_trainer import TrainModelPipeline

STAGE_NAME = "Data Ingestion"

try:
        logger.info(f"Stage: {STAGE_NAME} started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME_2  = "Data Preparation"

try:
        logger.info(f"Stage: {STAGE_NAME_2} started")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME_2} finished")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME_3 = "Base Model Preparation"

try:
        logger.info(f"Stage: {STAGE_NAME_3} started")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME_3} finished")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME_4 = "Model Training"
try:
        logger.info(f"Stage: {STAGE_NAME_4} started")
        obj = TrainModelPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME_4} finished")
except Exception as e:
        logger.exception(e)
        raise e