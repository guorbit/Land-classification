import constants
import pandas as pd
import sqlite3


def get_iteration(model_name):
    conn = sqlite3.connect("trial_parameters.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM trial_parameters WHERE model_name = ?", (model_name,)
    )
    count = cur.fetchone()[0]
    conn.close()
    return count


def write_to_database(
    hparams, model_name, model_iteration, accuracy, recall, precision, loss
):
    for key, value in hparams.items():
        if not (isinstance(value, str) or isinstance(value, float) or isinstance(value, int)):
            value = str(value)
    df = pd.DataFrame.from_dict(hparams, orient="index")
    df = df.transpose()
    df.insert(0, "model_name", model_name)
    df.insert(1, "model_iteration", model_iteration)
    df.insert(2, "accuracy", accuracy)
    df.insert(3, "recall", recall)
    df.insert(4, "precision", precision)
    df.insert(5, "loss_value", loss)

    # cast columns to string which are not str float or int
    for column in df.columns:
        if not (
            df[column].dtype == "float64"
            or df[column].dtype == "int64"
        ):
            df[column] = df[column].astype(str)

    

    conn = sqlite3.connect("trial_parameters.db")
    df.to_sql("trial_parameters", conn, if_exists="append", index=False)
    conn.close()


def init_db(hparams):
    for key, value in hparams.items():
        if not (isinstance(value, str) or isinstance(value, float) or isinstance(value, int)):
            value = str(value)
    df = pd.DataFrame.from_dict(hparams, orient="index")
    df = df.transpose()
    df.insert(0, "model_name", "")
    df.insert(1, "model_iteration", 0)
    df.insert(2, "accuracy", 0.0)
    df.insert(3, "recall", 0.0)
    df.insert(4, "precision", 0.0)
    df.insert(5, "loss_value", 0.0)

    # cast columns to string which are not str float or int
    for column in df.columns:
        if not (
            df[column].dtype == "float64"
            or df[column].dtype == "int64"
        ):
            df[column] = df[column].astype(str)

    df = df.drop(0)
    

    conn = sqlite3.connect("trial_parameters.db")
    df.to_sql("trial_parameters", conn, if_exists="append", index=False)
    conn.close()


if __name__ == "__main__":
    init_db(constants.HPARAMS)
