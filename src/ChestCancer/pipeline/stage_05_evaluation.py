import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import tensorflow as tf

# Must configure memory growth BEFORE any pipeline components initialize TF GPU
try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    pass

from ChestCancer import logger
from ChestCancer.config.configuration import ConfigManager
from ChestCancer.components.evaluation import Evaluation

STAGE_NAME = "Evaluation"


class EvaluationPipeline:

    def __init__(self):
        pass

    def main(self):
        config = ConfigManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"Stage: {STAGE_NAME} started")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e
