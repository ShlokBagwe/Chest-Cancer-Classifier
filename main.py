from ChestCancer.pipeline.stage_01_data_ingestion import DataIngestionPipeline 
from ChestCancer import logger
from ChestCancer.pipeline.stage_02_data_preparation import DataPreparationPipeline

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