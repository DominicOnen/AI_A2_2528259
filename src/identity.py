"""Student identity: derive the seed, file names and verification code."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class StudentIdentity:
    """Values derived from the institutional student ID."""

    student_id: str
    regdigits: str  # registration number without the slash, e.g. 2528259
    llll: int  # last four digits as an integer, used as the seed
    verification_code: str  # AI-A2-LLLL

    @property
    def dataset_name(self) -> str:
        return f"AI_A2_{self.regdigits}.csv"


def parse_student_id(student_id: str) -> StudentIdentity:
    """Parse an ID such as ``ug2528259@ines.ac.rw``.

    The digits of the part before ``@`` are the registration number without the
    slash. The last four digits form the seed (LLLL).
    """
    username = student_id.strip().split("@")[0]
    digits = re.sub(r"\D", "", username)
    if len(digits) < 5:
        raise ValueError(
            f"Student ID '{student_id}' must contain the registration digits, "
            "for example ug2528259@ines.ac.rw."
        )
    last_four = digits[-4:]
    return StudentIdentity(
        student_id=student_id.strip(),
        regdigits=digits,
        llll=int(last_four),
        verification_code=f"AI-A2-{last_four}",
    )


def check_seed(identity: StudentIdentity, seed: int) -> None:
    """Reject any seed that is not LLLL."""
    if seed != identity.llll:
        raise ValueError(
            f"Seed {seed} does not match your registration digits. "
            f"Expected {identity.llll} (last four digits of {identity.regdigits})."
        )
