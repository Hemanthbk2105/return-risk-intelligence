import pandas as pd
from pathlib import Path


INPUT_FILE = Path(
    "data/processed/model_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)


def main():

    print("=" * 70)
    print("RETURN-RISK INTELLIGENCE")
    print("TIME-BASED DATASET SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = pd.read_csv(
        INPUT_FILE
    )

    data["order_ts"] = pd.to_datetime(
        data["order_ts"]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = data.sort_values(
        "order_ts"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Find chronological boundaries
    # --------------------------------------------------------

    n = len(data)

    train_end = int(
        n * 0.60
    )

    validation_end = int(
        n * 0.80
    )

    train = data.iloc[
        :train_end
    ].copy()

    validation = data.iloc[
        train_end:validation_end
    ].copy()

    test = data.iloc[
        validation_end:
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    print("\nDataset split:")

    print(
        f"Total      : {len(data):,}"
    )

    print(
        f"Train      : {len(train):,}"
    )

    print(
        f"Validation : {len(validation):,}"
    )

    print(
        f"Test       : {len(test):,}"
    )

    print("\nDate ranges:")

    print(
        "Train:"
    )

    print(
        train["order_ts"].min(),
        "→",
        train["order_ts"].max()
    )

    print(
        "\nValidation:"
    )

    print(
        validation["order_ts"].min(),
        "→",
        validation["order_ts"].max()
    )

    print(
        "\nTest:"
    )

    print(
        test["order_ts"].min(),
        "→",
        test["order_ts"].max()
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print("\nReturn rates:")

    print(
        f"Train      : "
        f"{train['returned'].mean():.2%}"
    )

    print(
        f"Validation : "
        f"{validation['returned'].mean():.2%}"
    )

    print(
        f"Test       : "
        f"{test['returned'].mean():.2%}"
    )

    print("\nFiles created:")

    print("✓ train.csv")
    print("✓ validation.csv")
    print("✓ test.csv")

    print(
        "\nTIME-BASED SPLIT COMPLETE 🚀"
    )


if __name__ == "__main__":
    main()