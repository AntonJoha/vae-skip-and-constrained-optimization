import csv
import logging
from datetime import datetime, timezone
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_PATH = Path(__file__).with_name("data").joinpath("shampoo_sales.csv")


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise ValueError("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise ValueError("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise ValueError(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise ValueError(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise ValueError("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise ValueError("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise ValueError(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise ValueError(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                ).replace(tzinfo=timezone.utc)
                            else:
                                raise ValueError(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise ValueError("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise ValueError("Empty file.")
        if len(col_names) == 0:
            raise ValueError("Missing attribute section.")
        if not found_data_section:
            raise ValueError("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


class MaxMinDataset(Dataset):
    def __init__(self, series, context_length, horizon, max_val, min_val):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.context_length = context_length
        self.horizon = horizon
        max_val = torch.tensor(max_val, dtype=torch.float32)
        min_val = torch.tensor(min_val, dtype=torch.float32)
        self.diff = max_val - min_val

    def __len__(self):
        return max(0, len(self.series) - self.context_length - self.horizon + 1)

    def __getitem__(self, idx):

        x = self.series[idx : idx + self.context_length] / self.diff
        y = (
            self.series[
                idx + self.context_length : idx + self.context_length + self.horizon
            ]
            / self.diff
        )

        return x, y


class TimeSeriesDataset(Dataset):
    def __init__(self, series, context_length, horizon, mean, std):
        self.series = torch.tensor(series, dtype=torch.float32)
        self.context_length = context_length
        self.horizon = horizon

        # normalize the series
        # Compute statistics from the dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return max(0, len(self.series) - self.context_length - self.horizon + 1)

    def __getitem__(self, idx):
        x = (self.series[idx : idx + self.context_length] - self.mean) / self.std

        y = (
            self.series[
                idx + self.context_length : idx + self.context_length + self.horizon
            ]
            - self.mean
        ) / self.std

        return x, y


def _split_train_val_test(
    dataset: Dataset,
    train_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    if len(dataset) < 3:
        raise ValueError("dataset must contain at least three windows")

    # The clamp below keeps at least one window for validation and test.
    train_size = int(len(dataset) * train_fraction)
    train_size = min(max(1, train_size), len(dataset) - 2)
    remainder = len(dataset) - train_size
    val_size = max(1, remainder // 2)
    test_size = remainder - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )


def get_csv_dataset(
    file_path,
    context_length,
    horizon,
    batch_size=32,
    train_fraction=0.8,
    reduced_dataset=None,
    seed: int = 42,
):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, delimiter=";")

    if path.name == "AirQualityUCI.csv":
        # Drop the last two columns which are empty
        df = df.iloc[:, :-2]
        # Drop the original 'Date' and 'Time' columns
        df.drop(["Date", "Time"], axis=1, inplace=True)
        # Replace -200 values with NaN
        df.replace(-200, np.nan, inplace=True)
        # Forward fill NaN values
        df.ffill(inplace=True)
        # Replace commas with dots in object columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", ".", regex=False),
                    errors="coerce",
                )

    scaler = StandardScaler()
    df_norm = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index,
    )
    print(df.head())
    print(df_norm.head())

    dataset = WindowedSeriesDataset(
        torch.tensor(df_norm.values, dtype=torch.float32),
        seq_len=context_length,
        horizon=horizon,
    )

    return dataset


def _reduce_dataset(train_dataset, val_dataset, test_dataset, reduced_dataset, seed):
    train_size = max(1, int(reduced_dataset * len(train_dataset)))
    val_size = max(1, int(reduced_dataset * len(val_dataset)))
    test_size = max(1, int(reduced_dataset * len(test_dataset)))
    generator = torch.Generator().manual_seed(seed)
    train_dataset, _ = random_split(
        train_dataset,
        [train_size, len(train_dataset) - train_size],
        generator=generator,
    )
    val_dataset, _ = random_split(
        val_dataset,
        [val_size, len(val_dataset) - val_size],
        generator=generator,
    )
    test_dataset, _ = random_split(
        test_dataset,
        [test_size, len(test_dataset) - test_size],
        generator=generator,
    )
    return train_dataset, val_dataset, test_dataset


def get_tsf_dataset(
    tsf_file_path,
    context_length,
    horizon,
    batch_size=32,
    train_fraction=0.8,
    reduced_dataset=None,
    seed: int = 42,
):
    tsf_file_path = Path(tsf_file_path)
    if not tsf_file_path.exists():
        raise FileNotFoundError(tsf_file_path)
    df, _freq, _horizon, _has_missing, _equal_length = convert_tsf_to_dataframe(
        tsf_file_path
    )

    ## normalize data
    all_values = []
    print(df.head())

    seen = []
    station_ids = {}
    for station, obs in zip(df["station_id"], df["obs_or_fcst"], strict=False):
        if station not in seen:
            seen.append(station)
            station_ids[station] = []
            print("Station_id", station)
        station_ids[station].append(obs)

    for station in seen:
        print("Station_id", station, "Obs_or_fcst", station_ids[station])

    for row in df["series_value"]:
        all_values.extend(row)
    all_values = np.array(all_values)
    mean = np.mean(all_values)
    std = np.std(all_values)

    dataset = None
    for row in df["series_value"]:
        if dataset is None:
            dataset = TimeSeriesDataset(
                row, context_length=context_length, horizon=horizon, mean=mean, std=std
            )
        else:
            dataset = ConcatDataset(
                [
                    dataset,
                    TimeSeriesDataset(
                        row,
                        context_length=context_length,
                        horizon=horizon,
                        mean=mean,
                        std=std,
                    ),
                ]
            )

    return dataset


class WindowedSeriesDataset(Dataset):
    def __init__(self, series: torch.Tensor, seq_len: int, horizon: int = 1) -> None:
        if len(series) <= seq_len:
            raise ValueError("series must be longer than seq_len")

        self.series = series
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self) -> int:
        return max(0, len(self.series) - self.seq_len - self.horizon + 1)

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence = self.series[index : index + self.seq_len]
        target = self.series[index + self.seq_len : index + self.seq_len + self.horizon]
        return sequence, target


def load_series(path: Path) -> torch.Tensor:
    values = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values.append(float(row["sales"]))

    series = torch.tensor(values, dtype=torch.float32)
    mean = series.mean()
    std = series.std(unbiased=False).clamp_min(1e-6)
    return (series - mean) / std


def get_shampoo_dataloaders(
    config,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    series = load_series(DATASET_PATH)
    dataset = WindowedSeriesDataset(series, config.seq_len, config.horizon)

    train_dataset, val_dataset, test_dataset = _split_train_val_test(
        dataset,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )

    if config.reduced_dataset is not None:
        train_size = max(1, int(config.reduced_dataset * len(train_dataset)))
        val_size = max(1, int(config.reduced_dataset * len(val_dataset)))
        test_size = max(1, int(config.reduced_dataset * len(test_dataset)))
        generator = torch.Generator().manual_seed(config.seed)
        train_dataset, _ = random_split(
            train_dataset,
            [train_size, len(train_dataset) - train_size],
            generator=generator,
        )
        val_dataset, _ = random_split(
            val_dataset,
            [val_size, len(val_dataset) - val_size],
            generator=generator,
        )
        test_dataset, _ = random_split(
            test_dataset,
            [test_size, len(test_dataset) - test_size],
            generator=generator,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def _ped_get_mean_std(train):

    arr = []

    for files in [list(train.glob("*.txt"))]:
        for file in files:
            with open(file) as f:
                for line in f:
                    values = line.split()[2:]  # keep columns 3 and onward
                    arr.append(np.array(values))

    arr = np.array(arr, dtype=np.float32)
    print("arr shape", arr.shape)
    max_val = np.max(arr, axis=0)
    min_val = np.min(arr, axis=0)

    return max_val, min_val


def _ped_get_folder(ped_folder, context_length, horizon, max_val, min_val):

    files = list(ped_folder.glob("*.txt"))

    array_of_datasets = []

    for file in files:
        curr = []
        with open(file) as f:
            for line in f:
                values = line.split()[2:]  # keep columns 3 and onward
                curr.append(np.array(values, dtype=np.float32))
        array_of_datasets.append(
            MaxMinDataset(
                curr,
                context_length=context_length,
                horizon=horizon,
                max_val=max_val,
                min_val=min_val,
            )
        )

    return ConcatDataset(array_of_datasets)


def get_ped_dataset(
    ped_file_path,
    context_length,
    horizon,
    batch_size=32,
    train_fraction=0.8,
    reduced_dataset=None,
    seed: int = 42,
):

    ped_file_path = ped_file_path.with_suffix("")

    ped_train = ped_file_path / "train"
    ped_test = ped_file_path / "test"
    ped_val = ped_file_path / "val"

    max_val, min_val = _ped_get_mean_std(ped_train)

    ped_train_dataset = _ped_get_folder(
        ped_train, context_length, horizon, max_val, min_val
    )
    pred_test_dataset = _ped_get_folder(
        ped_test, context_length, horizon, max_val, min_val
    )
    ped_val_dataset = _ped_get_folder(
        ped_val, context_length, horizon, max_val, min_val
    )

    test = DataLoader(pred_test_dataset, batch_size=batch_size, shuffle=False)
    train = DataLoader(ped_train_dataset, batch_size=batch_size, shuffle=True)
    val = DataLoader(ped_val_dataset, batch_size=batch_size, shuffle=False)

    return train, val, test


def make_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_df, val_df, test_df = _make_dataloaders(config)

    print("Train dataset len", len(train_df))
    print("Val dataset len", len(val_df))
    print("Test dataset len", len(test_df))
    return train_df, val_df, test_df


def _make_dataloaders(config) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_path = Path(get_dataset_names()[0])
    print(f"Loading dataset from {dataset_path}")

    dataset = None

    if dataset_path.suffix == ".ped":
        return get_ped_dataset(
            dataset_path,
            config.seq_len,
            config.horizon,
            config.batch_size,
            train_fraction=config.train_fraction,
            reduced_dataset=config.reduced_dataset,
            seed=getattr(config, "seed", 42),
        )

    if dataset_path.suffix == ".tsf":
        dataset = get_tsf_dataset(
            dataset_path,
            config.seq_len,
            config.horizon,
            config.batch_size,
            train_fraction=config.train_fraction,
            reduced_dataset=config.reduced_dataset,
            seed=getattr(config, "seed", 42),
        )

    if dataset_path.suffix == ".csv":
        dataset = get_csv_dataset(
            dataset_path,
            config.seq_len,
            config.horizon,
            config.batch_size,
            train_fraction=config.train_fraction,
            reduced_dataset=config.reduced_dataset,
            seed=getattr(config, "seed", 42),
        )

    train_dataset, val_dataset, test_dataset = _split_train_val_test(
        dataset,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )

    if config.reduced_dataset is not None:
        train_dataset, val_dataset, test_dataset = _reduce_dataset(
            train_dataset,
            val_dataset,
            test_dataset,
            config.reduced_dataset,
            config.seed,
        )

    train_df = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_df = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_df = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    logger.info(f"train dataset size: {len(train_dataset)}")
    logger.info(f"val dataset size: {len(val_dataset)}")
    logger.info(f"test dataset size: {len(test_dataset)}")
    return train_df, val_df, test_df


def ped_rescale(series, max_val, min_val):
    diff = max_val - min_val
    return (series - min_val) / diff


def ped_get_min_max(path):

    ped_file_path = path.with_suffix("")
    ped_train = ped_file_path / "train"
    max_val, min_val = _ped_get_mean_std(ped_train)

    return max_val, min_val


def get_scale_constant(runtime):
    """
    Returns lambda that redo the normalization of the data. This is used to scale the output of the model back to the original scale.
    """

    dataset_path = Path(get_dataset_names()[0])
    print(f"Loading dataset from {dataset_path}")

    if dataset_path.suffix == ".ped":
        max_val, min_val = ped_get_min_max(dataset_path)
        diff = max_val - min_val
        print("MAX", max_val, "MIN", min_val, "DIFF", diff)
        return lambda x: x * diff, lambda x: x + 2 * np.log(diff)

    else:
        # TODO : Implement scaling for other dataset types
        return lambda x: x


def get_dataset_names():
    return [
        "data/eth.ped",
        "data/covid_deaths_dataset.tsf",
        "data/AirQualityUCI.csv",
        "data/pedestrian_counts_dataset.tsf",
        "data/solar_10_minutes_dataset.tsf",
        "data/m1_monthly_dataset.tsf",
        "data/traffic_weekly_dataset.tsf",
    ]
