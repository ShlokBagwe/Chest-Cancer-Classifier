from ChestCancer import logger
from ChestCancer.config.configuration import ConfigManager
from ChestCancer.components.data_preparation import DataPreparation

STAGE_NAME = "Data Preparation"

class DataPreparationPipeline:

    def __init__(self):
        pass

    def main(self):

        try:
            config = ConfigManager()
            config_data_preparation = config.get_data_preparation()
            data_preparation  = DataPreparation(config_data_preparation)
            data_preparation.prepare_all()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"Stage {STAGE_NAME} started")
        
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
