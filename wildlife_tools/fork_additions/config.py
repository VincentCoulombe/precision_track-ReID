import os

from pydantic import BaseModel, Field, field_validator, model_validator

from .nn import BACKBONE
from .utils import VIDEO_EXTENSIONS

MOT_PHASES = ("train", "val")

EMPTY_PHASE_EXPLANATIONS = {
    "train": (
        "the model has no examples to learn identities from. A re-identification network learns its "
        "embedding space entirely from the training crops, so an empty training set can't produce a "
        "meaningful model at all."
    ),
    "val": (
        "there is no held-out data to evaluate on. Without a validation split, you can't measure how well "
        "the model generalizes to identities/frames it wasn't trained on, catch overfitting during training, "
        "or trust the reported F1 scores and calibrated confidence at deploy time (training on 100% of your "
        "data and reporting metrics on that same data silently overstates real-world performance)."
    ),
}


class UserConfig(BaseModel):
    train: bool
    test: bool
    deploy: bool

    num_classes: int = Field(gt=0)

    dataset_directory: str
    metadata: str
    save_directory: str

    confidence_threshold: float = Field(ge=0.0, le=1.0)
    bbox_enlargement: float = Field(ge=0.0)

    backbone_name: str
    freeze_backbone: bool
    batch_size: int = Field(ge=1, le=160)
    epochs: int = Field(gt=0)
    val_split: float = Field(gt=0.0, lt=1.0)
    seed: int

    @field_validator("save_directory")
    @classmethod
    def save_directory_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("save_directory must not be blank.")
        return v

    @field_validator("backbone_name")
    @classmethod
    def backbone_name_known(cls, v: str) -> str:
        if v.lower() not in BACKBONE:
            raise ValueError(f"Unknown backbone_name: '{v}'. Available: {list(BACKBONE.keys())}.")
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_size_divides_accumulation(cls, v: int) -> int:
        if 160 // v == 0:
            raise ValueError(f"batch_size must be <= 160 (160 // batch_size must be >= 1), got {v}.")
        return v

    @model_validator(mode="after")
    def dataset_paths_exist(self) -> "UserConfig":
        if not os.path.isdir(self.dataset_directory):
            raise ValueError(f"dataset_directory does not exist: '{self.dataset_directory}'.")
        for sub in ("bboxes", "videos"):
            sub_path = os.path.join(self.dataset_directory, sub)
            if not os.path.isdir(sub_path):
                raise ValueError(f"dataset_directory is missing the '{sub}' sub directory: '{sub_path}'.")
        metadata_path = os.path.join(self.dataset_directory, self.metadata)
        if not os.path.isfile(metadata_path):
            raise ValueError(f"metadata file does not exist: '{metadata_path}'.")

        errors = []
        for phase in MOT_PHASES:
            bboxes_phase_dir = os.path.join(self.dataset_directory, "bboxes", phase)
            videos_phase_dir = os.path.join(self.dataset_directory, "videos", phase)

            if not os.path.isdir(bboxes_phase_dir):
                errors.append(f"missing bboxes/{phase} directory: '{bboxes_phase_dir}'.")
            if not os.path.isdir(videos_phase_dir):
                errors.append(f"missing videos/{phase} directory: '{videos_phase_dir}'.")
            if not os.path.isdir(bboxes_phase_dir) or not os.path.isdir(videos_phase_dir):
                continue

            video_files = [f for f in os.listdir(videos_phase_dir) if f.lower().endswith(VIDEO_EXTENSIONS)]
            if not video_files:
                errors.append(
                    f"videos/{phase} contains no video files ({videos_phase_dir}) — {EMPTY_PHASE_EXPLANATIONS[phase]}"
                )
                continue

            for video_file in video_files:
                stem = os.path.splitext(video_file)[0]
                expected_csv = os.path.join(bboxes_phase_dir, f"{stem}.csv")
                if not os.path.isfile(expected_csv):
                    errors.append(f"videos/{phase}/{video_file} has no matching bboxes/{phase}/{stem}.csv.")

        if errors:
            raise ValueError("Invalid MOT dataset structure:\n" + "\n".join(f"  - {e}" for e in errors))

        return self
