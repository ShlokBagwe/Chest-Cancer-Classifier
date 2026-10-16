import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from ChestCancer import logger
from ChestCancer.config.configuration import ConfigManager
from ChestCancer.components.model_trainer import Training

STAGE_NAME = "Train Model"


class TrainModelPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        training_config = config.get_training_config()
        training = Training(training_config)
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f"Stage: {STAGE_NAME} started")
        obj = TrainModelPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
