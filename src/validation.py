from src.helpers.logger import logger
from typing import List, Tuple

def validate_scores(scores: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    valid_scores = []

    for score, desc in scores:
        if score < 0 or score > 60:
            logger.warning(f"Filtering out impossible score: {score}")
            continue

        if score > 60 and not any(x in desc.upper() for x in ["T"]):
            logger.warning(f"High score {score} without triple indication: {desc}")
            continue

        if len(valid_scores) >= 3:
            logger.warning("Already 3 scores")
            break

        valid_scores.append((score, desc))

    return valid_scores