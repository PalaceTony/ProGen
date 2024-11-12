import random
import numpy as np
import torch
import logging


def get_adjacency_matrix2(
    distance_df_filename, num_of_vertices, type_="connectivity", id_filename=None
):
    """
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    """
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, "r") as f:
            id_dict = {
                int(i): idx for idx, i in enumerate(f.read().strip().split("\n"))
            }
        with open(distance_df_filename, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == "connectivity":
                A[i, j] = 1
                # A[j, i] = 1
            elif type == "distance":
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be " "connectivity or distance!")
    return A


def seed_everything(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


local_rank = 0


def set_local_rank(rank):
    global local_rank
    local_rank = rank


def get_local_rank():
    return local_rank


def gather(consts: torch.Tensor, t: torch.Tensor):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)


import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    logger = logging.getLogger("score_sde")

    # Define a function to log uncaught exceptions
    def log_uncaught_exceptions(exctype, value, traceback):
        logger.error("Uncaught exception", exc_info=(exctype, value, traceback))

    # Assign the function to sys.excepthook
    sys.excepthook = log_uncaught_exceptions


import torch
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state["optimizer"].load_state_dict(loaded_state["optimizer"])
    state["model"].load_state_dict(loaded_state["model"], strict=False)
    state["ema"].load_state_dict(loaded_state["ema"])
    state["step"] = loaded_state["step"]
    return state


def save_checkpoint(ckpt_dir, state):
    # Ensure the directory exists
    saved_state = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": state["step"],
    }
    torch.save(saved_state, ckpt_dir)


import pandas as pd
import os


def write_metrics_to_excel(
    metrics,
    notes=None,
    log_dir=None,
    file_path="outputs/sampling_metrics.xlsx",
):
    # Place notes and log_dir as the first columns
    columns = ["Notes", "LogDir", "MAE", "RMSE", "MAPE", "CRPS", "MIS"]
    metrics = [
        notes,
        log_dir,
    ] + metrics  # Prepend notes and log_dir to the metrics list
    metrics_df = pd.DataFrame([metrics], columns=columns)

    if os.path.exists(file_path):
        # Append to the existing file
        with pd.ExcelWriter(file_path, mode="a", if_sheet_exists="overlay") as writer:
            startrow = writer.sheets["Sheet1"].max_row
            metrics_df.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        # Write a new file
        metrics_df.to_excel(file_path, index=False)


def write_train_metrics_to_excel(
    metrics,
    notes=None,
    log_dir=None,
    file_path="outputs/training_logs.xlsx",
):
    # Place notes and log_dir as the first columns
    columns = [
        "Notes",
        "LogDir",
        "TotalTrainingTime",
        "BestValLoss",
        "BestEpochNumber",
    ]
    metrics = [
        notes,
        log_dir,
    ] + metrics  # Prepend notes and log_dir to the metrics list
    metrics_df = pd.DataFrame([metrics], columns=columns)

    if os.path.exists(file_path):
        # Append to the existing file
        with pd.ExcelWriter(file_path, mode="a", if_sheet_exists="overlay") as writer:
            startrow = writer.sheets["Sheet1"].max_row
            metrics_df.to_excel(writer, index=False, header=False, startrow=startrow)
    else:
        # Write a new file
        metrics_df.to_excel(file_path, index=False)


import os
import pandas as pd


def write_all_to_excel(
    train_metrics=None,
    sample_metrics=None,
    best_sample_metrics=None,
    notes=None,
    log_dir=None,
    file_path="outputs/metrics.xlsx",
    update_existing=True,
):
    # Define columns
    columns = [
        "Notes",
        "LogDir",
        "TotalTrainingTime",
        "BestValLoss",
        "BestEpochNumber",
        "MAE",
        "RMSE",
        "MAPE",
        "CRPS",
        "MIS",
        "MAE_Best",
        "RMSE_Best",
        "MAPE_Best",
        "CRPS_Best",
        "MIS_Best",
    ]

    # Initialize metrics with NAs
    metrics = {
        "Notes": notes,
        "LogDir": log_dir,
        "TotalTrainingTime": "NA",
        "BestValLoss": "NA",
        "BestEpochNumber": "NA",
        "MAE": "NA",
        "RMSE": "NA",
        "MAPE": "NA",
        "CRPS": "NA",
        "MIS": "NA",
        "MAE_Best": "NA",
        "RMSE_Best": "NA",
        "MAPE_Best": "NA",
        "CRPS_Best": "NA",
        "MIS_Best": "NA",
    }

    # Update metrics if provided
    if train_metrics is not None:
        metrics.update(
            {
                "TotalTrainingTime": train_metrics[0],
                "BestValLoss": train_metrics[1],
                "BestEpochNumber": train_metrics[2],
            }
        )

    if sample_metrics is not None:
        metrics.update(
            {
                "MAE": sample_metrics[0],
                "RMSE": sample_metrics[1],
                "MAPE": sample_metrics[2],
                "CRPS": sample_metrics[3],
                "MIS": sample_metrics[4],
            }
        )

    if best_sample_metrics is not None:
        metrics.update(
            {
                "MAE_Best": best_sample_metrics[0],
                "RMSE_Best": best_sample_metrics[1],
                "MAPE_Best": best_sample_metrics[2],
                "CRPS_Best": best_sample_metrics[3],
                "MIS_Best": best_sample_metrics[4],
            }
        )

    # Convert to DataFrame
    metrics_df = pd.DataFrame([metrics], columns=columns)

    # Write to Excel
    if os.path.exists(file_path):
        # Read existing file
        existing_df = pd.read_excel(file_path)

        if update_existing:
            # Check if the row already exists based on "Notes"
            match_idx = existing_df[existing_df["Notes"] == notes].index

            if not match_idx.empty:
                # Update the matching row while keeping "Notes" column unchanged
                for idx in match_idx:
                    for col in columns:
                        if col != "Notes":
                            existing_df.at[idx, col] = metrics[col]
            else:
                # Append new row
                existing_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        else:
            # Append new row
            existing_df = pd.concat([existing_df, metrics_df], ignore_index=True)

        with pd.ExcelWriter(file_path, mode="w") as writer:
            existing_df.to_excel(writer, index=False)
    else:
        # Write a new file
        metrics_df.to_excel(file_path, index=False)


import sqlite3
import pandas as pd
import numpy as np


def write_all_to_sqlite(
    train_metrics=None,
    sample_metrics=None,
    best_sample_metrics=None,
    notes=None,
    log_dir=None,
    db_path="outputs/metrics.db",
    table_name="metrics",
    update_existing=True,
):
    # Define columns and types
    columns = [
        "Notes TEXT",
        "LogDir TEXT",
        "TotalTrainingTime REAL",
        "BestValLoss REAL",
        "BestEpochNumber INTEGER",
        "MAE REAL",
        "RMSE REAL",
        "MAPE REAL",
        "CRPS REAL",
        "MIS REAL",
        "MAE_Best REAL",
        "RMSE_Best REAL",
        "MAPE_Best REAL",
        "CRPS_Best REAL",
        "MIS_Best REAL",
    ]

    # Initialize metrics with NAs
    metrics = {
        "Notes": notes,
        "LogDir": log_dir,
        "TotalTrainingTime": np.nan,
        "BestValLoss": np.nan,
        "BestEpochNumber": np.nan,
        "MAE": np.nan,
        "RMSE": np.nan,
        "MAPE": np.nan,
        "CRPS": np.nan,
        "MIS": np.nan,
        "MAE_Best": np.nan,
        "RMSE_Best": np.nan,
        "MAPE_Best": np.nan,
        "CRPS_Best": np.nan,
        "MIS_Best": np.nan,
    }

    # Update metrics if provided
    if train_metrics is not None:
        metrics.update(
            {
                "TotalTrainingTime": train_metrics[0],
                "BestValLoss": train_metrics[1],
                "BestEpochNumber": train_metrics[2],
            }
        )

    if sample_metrics is not None:
        metrics.update(
            {
                "MAE": float(sample_metrics[0]),
                "RMSE": float(sample_metrics[1]),
                "MAPE": float(sample_metrics[2]),
                "CRPS": float(sample_metrics[3]),
                "MIS": float(sample_metrics[4]),
            }
        )

    if best_sample_metrics is not None:
        metrics.update(
            {
                "MAE_Best": float(best_sample_metrics[0]),
                "RMSE_Best": float(best_sample_metrics[1]),
                "MAPE_Best": float(best_sample_metrics[2]),
                "CRPS_Best": float(best_sample_metrics[3]),
                "MIS_Best": float(best_sample_metrics[4]),
            }
        )

    # Convert to DataFrame
    metrics_df = pd.DataFrame([metrics])

    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # Create table if it doesn't exist
    conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})")

    if update_existing:
        # Check if the row already exists based on "Notes"
        query = f"SELECT rowid FROM {table_name} WHERE Notes = ?"
        match_idx = conn.execute(query, (notes,)).fetchone()

        if match_idx:
            # Update the matching row
            update_query = f"UPDATE {table_name} SET {', '.join([f'{col.split()[0]} = ?' for col in columns if 'Notes' not in col])} WHERE Notes = ?"
            values_to_update = list(metrics.values())[1:] + [
                notes
            ]  # Skip "Notes" value for updating
            conn.execute(update_query, values_to_update)
        else:
            # Insert new row
            metrics_df.to_sql(table_name, conn, if_exists="append", index=False)
    else:
        # Insert new row
        metrics_df.to_sql(table_name, conn, if_exists="append", index=False)

    # Commit and close connection
    conn.commit()
    conn.close()


import shutil
import os


def collect_code_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".py"):
                # Get the relative path to preserve directory structure
                rel_path = os.path.relpath(root, source_dir)
                dest_path = os.path.join(destination_dir, rel_path)

                # Create directories in the destination path if they don't exist
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)

                full_file_path = os.path.join(root, file)
                shutil.copy(full_file_path, dest_path)
