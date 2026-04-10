"""CLI script to create the optional synthetic course-intent dataset."""

from pathlib import Path

from src.data_utils import create_synthetic_dataset


if __name__ == "__main__":
    output_path = Path("data/raw/intent_dataset.csv")
    df = create_synthetic_dataset(output_path)
    print(f"Dataset created at: {output_path}")
    print("Shape:", df.shape)
    print("Class counts:")
    print(df["intent"].value_counts())
