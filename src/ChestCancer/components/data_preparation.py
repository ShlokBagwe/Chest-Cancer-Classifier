from numpy import split
from pandas._libs import missing
import shutil
from ChestCancer import logger
from pathlib import Path
from ChestCancer.entity.config_entity import DataPreparationConfig
import pandas as pd


class DataPreparation:

    def __init__(self, config: DataPreparationConfig):
        self.config  = config

    def read_csv(self, split: str) -> pd.DataFrame:
        return pd.read_csv(self.config.data_dir / split / "_classes.csv")
    
    
    def add_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        raw_types = df.iloc[:, 1:].idxmax(axis=1)

        """
            converting everything into 4 classes
            eg:
            adenocarcinoma -> adenocarcinoma 
            large.cell -> large_cell 
            squamous.cell -> squamous_cell
        """
        def clean_label(val: str) -> str:
            for prefix, target in self.config.label_prefixes.items():
                if val.startswith(prefix):
                    return target
            raise ValueError(f"No label prefix matches column: {val}")

        df["type"] = raw_types.apply(clean_label)
        return df


    def split_data_dir(self, split: str, df: pd.DataFrame):
        """
        Splitting data into its different type of disease but preserving the split of the data ie train, test and valid
        """

        data_src = Path(self.config.data_dir)
        data_dst = Path(self.config.prepared_dir)

        missing = []
        for _,row in df.iterrows():

            src = data_src / split / row['filename']
            dst = data_dst / split / row['type'] / row['filename']

            if not src.exists():
                missing.append(row['filename'])
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

        if missing:
            logger.warning(f"Split '{split}': {len(missing)} files were missing and skipped.")
        else:
            logger.info(f"Split '{split}': all {len(df)} files successfully copied to {data_dst / split}")


    def prepare_all(self):
        for split in self.config.splits:
            logger.info(f"Preparing data for split: {split}")
            df = self.read_csv(split)
            df = self.add_type(df)
            self.split_data_dir(split, df)

