from ChestCancer.pipeline.stage_01_data_ingestion import DataIngestionPipeline 
from ChestCancer import logger

STAGE_NAME = "Data Ingestion"

try:
        logger.info(f"Stage: {STAGE_NAME} started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
except Exception as e:
        logger.exception(e)
        raise e