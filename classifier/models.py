"""Prompt-shaped inputs. Imported assignments are never training labels."""

from datetime import date
from decimal import Decimal
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TimeEntry(StrictModel):
    UserId: str = Field(min_length=1, max_length=256)
    EntryId: str = Field(min_length=1, max_length=256)
    Narrative: str = Field(min_length=1, max_length=16000)
    Hours: Decimal = Field(ge=0, le=1_000_000, decimal_places=9)
    ClientName: str = Field(max_length=2000)
    MatterName: str = Field(max_length=2000)
    EntryDate: date

    @field_validator("UserId", "EntryId", "Narrative")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value

    def record(self):
        value = self.model_dump(mode="json")
        value["Hours"] = float(self.Hours)
        return value


class Matter(StrictModel):
    ClientId: str = Field(min_length=1, max_length=256)
    ClientName: str = Field(min_length=1, max_length=2000)
    MatterId: str = Field(min_length=1, max_length=256)
    MatterName: str = Field(min_length=1, max_length=2000)
    MatterDescription: str = Field(max_length=16000)
    PracticeArea: str = ""
    MatterType: str = ""
    Status: Literal["Open", "Closed"] = "Open"
    RelatedKeywords: list[str] = Field(default_factory=list, max_length=100)
    InvolvedTimekeepers: list[str] = Field(default_factory=list, max_length=1000)

    @field_validator("ClientId", "ClientName", "MatterId", "MatterName")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value

    def text(self):
        return " ".join(
            [
                self.ClientName,
                self.MatterName,
                self.MatterDescription,
                " ".join(self.RelatedKeywords),
                self.PracticeArea,
                self.MatterType,
            ]
        )


class Decision(StrictModel):
    revision: int = Field(ge=1)
    action: Literal["accept", "reject", "undo", "retry"]
    candidate_id: str | None = None


class Edit(StrictModel):
    revision: int = Field(ge=1)
    input: TimeEntry
    reset_decision: bool = False
