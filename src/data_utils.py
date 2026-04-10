"""Dataset loading and splitting utilities."""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import FALLBACK_RAW_DATA_PATH, RANDOM_SEED, RAW_DATA_PATH


TEXT_CANDIDATE_COLUMNS = ("text", "instruction", "query", "utterance", "message")
INTENT_CANDIDATE_COLUMNS = ("intent", "label", "category")
RESPONSE_CANDIDATE_COLUMNS = ("response", "reply", "answer")


def create_synthetic_dataset(save_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Create a balanced intent dataset using realistic templates."""
    courses = ["NLP", "Machine Learning", "Data Structures"]
    greetings = [
        "Hi",
        "Hello",
        "Good morning",
        "Hey there",
        "Hi assistant",
        "Hello professor bot",
        "Good evening",
        "Hey",
        "Hi, can you help me?",
        "Hello, I have a question",
        "Hey assistant",
        "Hi chatbot",
    ]
    goodbyes = [
        "Bye",
        "Goodbye",
        "See you later",
        "Thanks, bye",
        "That is all, goodbye",
        "Talk to you later",
        "Catch you later",
        "I am done, bye",
        "See you",
        "Have a nice day, bye",
        "Thanks for the help, bye",
        "Okay, goodbye",
    ]

    rows = []
    for c in courses:
        rows.extend(
            [
                (f"{g}!", "greeting") for g in greetings
            ]
        )
        rows.extend(
            [
                (f"What topics are covered in {c} this semester?", "course_info"),
                (f"Can you share the syllabus for {c}?", "course_info"),
                (f"How many credits is {c}?", "course_info"),
                (f"Is attendance required for {c}?", "course_info"),
                (f"What is the grading policy in {c}?", "course_info"),
                (f"Which textbook do we use in {c}?", "course_info"),
                (f"Do we have a project in {c}?", "course_info"),
                (f"What are the weekly topics in {c}?", "course_info"),
                (f"Is {c} an elective or compulsory course?", "course_info"),
                (f"How is participation graded in {c}?", "course_info"),
                (f"Where can I find materials for {c}?", "course_info"),
                (f"What is the course description for {c}?", "course_info"),
            ]
        )
        rows.extend(
            [
                (f"When is assignment 1 due for {c}?", "assignment_deadline"),
                (f"What is the deadline for homework 2 in {c}?", "assignment_deadline"),
                (f"Could you remind me of the project submission date for {c}?", "assignment_deadline"),
                (f"Is the lab report deadline this week for {c}?", "assignment_deadline"),
                (f"When should we submit the final project in {c}?", "assignment_deadline"),
                (f"What time is the assignment deadline for {c}?", "assignment_deadline"),
                (f"Do we have an extension for the last assignment in {c}?", "assignment_deadline"),
                (f"What is the due date of week 3 homework in {c}?", "assignment_deadline"),
                (f"Please tell me the next assignment deadline for {c}", "assignment_deadline"),
                (f"Deadline for the coding task in {c}?", "assignment_deadline"),
                (f"When do we upload the take-home task for {c}?", "assignment_deadline"),
                (f"I forgot the submission date for {c} assignment", "assignment_deadline"),
            ]
        )
        rows.extend(
            [
                (f"When is the midterm exam for {c}?", "exam_date"),
                (f"What date is the final exam in {c}?", "exam_date"),
                (f"Do you know the quiz date for {c}?", "exam_date"),
                (f"Has the exam schedule for {c} been announced?", "exam_date"),
                (f"Can you tell me the exam day for {c}?", "exam_date"),
                (f"What time does the {c} midterm start?", "exam_date"),
                (f"Is the final of {c} in January?", "exam_date"),
                (f"When will the makeup exam for {c} be held?", "exam_date"),
                (f"Where can I see the exam calendar for {c}?", "exam_date"),
                (f"Exam date update for {c} please", "exam_date"),
                (f"When is our first quiz in {c}?", "exam_date"),
                (f"Did the professor change the exam date of {c}?", "exam_date"),
            ]
        )
        rows.extend(
            [
                (f"When are your office hours for {c}?", "office_hours"),
                (f"Can I meet the instructor for {c} tomorrow?", "office_hours"),
                (f"What time can we visit during office hours for {c}?", "office_hours"),
                (f"Where is the office for {c} consultation?", "office_hours"),
                (f"Do you hold online office hours for {c}?", "office_hours"),
                (f"How can I book an office hour slot for {c}?", "office_hours"),
                (f"Are office hours available on Friday for {c}?", "office_hours"),
                (f"Could you share the office location for {c}?", "office_hours"),
                (f"Is there a Zoom link for {c} office hour?", "office_hours"),
                (f"I need to discuss grades, when are office hours for {c}?", "office_hours"),
                (f"What are the consultation hours for {c}?", "office_hours"),
                (f"Can I ask questions after class for {c} office hours?", "office_hours"),
            ]
        )
        rows.extend(
            [
                (f"{g}.", "goodbye") for g in goodbyes
            ]
        )

    df = pd.DataFrame(rows, columns=["text", "intent"])
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    return df


def _find_first_existing_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"None of the required columns exist: {candidates}")


def normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw dataset to standard columns: text, intent, response(optional)."""
    text_col = _find_first_existing_column(df, TEXT_CANDIDATE_COLUMNS)
    intent_col = _find_first_existing_column(df, INTENT_CANDIDATE_COLUMNS)

    normalized = pd.DataFrame()
    normalized["text"] = df[text_col].astype(str).str.strip()
    normalized["intent"] = df[intent_col].astype(str).str.strip()

    response_col = None
    for col in RESPONSE_CANDIDATE_COLUMNS:
        if col in df.columns:
            response_col = col
            break
    if response_col:
        normalized["response"] = df[response_col].astype(str).str.strip()

    normalized = normalized[(normalized["text"] != "") & (normalized["intent"] != "")]
    normalized = normalized.dropna(subset=["text", "intent"]).reset_index(drop=True)
    return normalized


def load_dataset(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load dataset from CSV and normalize to text/intent format."""
    if csv_path.exists():
        raw_df = pd.read_csv(csv_path)
        return normalize_dataset_columns(raw_df)
    if FALLBACK_RAW_DATA_PATH.exists():
        raw_df = pd.read_csv(FALLBACK_RAW_DATA_PATH)
        return normalize_dataset_columns(raw_df)
    return create_synthetic_dataset(FALLBACK_RAW_DATA_PATH)


def build_intent_response_map(df: pd.DataFrame) -> Dict[str, str]:
    """Create intent -> representative response mapping."""
    if "response" in df.columns:
        sample_rows = (
            df[df["response"].astype(str).str.len() > 0]
            .drop_duplicates(subset=["intent"])
            .loc[:, ["intent", "response"]]
        )
        return dict(zip(sample_rows["intent"], sample_rows["response"]))
    return {}


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into stratified train/validation/test sets."""
    if df["intent"].nunique() < 2:
        raise ValueError("Dataset must contain at least two distinct intent classes.")

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["intent"],
    )
    adjusted_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        random_state=random_seed,
        stratify=train_val["intent"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
