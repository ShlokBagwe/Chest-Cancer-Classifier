from ChestCancer import logger
from ChestCancer.config.configuration import ConfigManager
from ChestCancer.components.prepare_baseline_model import PrepareBaselineModel

STAGE_NAME = "Base Model"


class PrepareBaseModelPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        prepare_base_model_config = config.get_baseline_model_config()
        prepare_base_model = PrepareBaselineModel(prepare_base_model_config)
        prepare_base_model.get_baseline_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"Stage: {STAGE_NAME} started")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
