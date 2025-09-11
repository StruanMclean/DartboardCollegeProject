
from typing import List, Tuple
from helpers.logger import logger

def filter_reasonable_scores(scores: List[tuple[int, str]]) -> List[tuple[int, str]]:
    vaild_scores = []

    for score, desc in scores:
        if score < 0 or score > 60:
            continue

        if score > 60 and not any(x in desc.upper() for x in ['T', 'TRIPLE']):
            continue

        vaild_scores.append((score, desc))
    
    return vaild_scores

def compare_duo_cam(
    scores1: List[Tuple[int, str]], 
    scores2: List[Tuple[int, str]], 
    confidence_threshold: float = 0.6
) -> List[Tuple[int, str]]:
    """
    Checking both cams for accuracy
    i think my spelling is right
    lowkey
    """

    if not scores1 and not scores2:
        return []
    
    if not scores1:
        logger.warning("Camera 1 detected no darts, using Camera 2 results")
        return scores2
    
    if not scores2:
        logger.warning("Camera 2 detected no darts, using Camera 1 results")
        return scores1
    
    # Match
    exact_matches = []

    scores1_set = set(scores1)
    scores2_set = set(scores2)

    for score in scores1_set.intersection(scores2_set):
        exact_matches.append(score)

    # Prefer more pesific
    score_matches = []

    for i, score1 in enumerate(scores1):
        for i, score2 in enumerate(scores2):
            if (
                score1[0] == score2[0] and
                score1 not in exact_matches and 
                score2 not in exact_matches
            ):
                better_descriptor = score1[1] if len(score[1]) > len(score2) else score2[1]
                score_matches.append((score1[0], better_descriptor))
                break

    validated_scores = exact_matches + score_matches

    # Significant dissagrement
    total_detected = len(scores1) + len(scores2)
    agreement_ratio = len(validated_scores) / max(total_detected, 1)

    if agreement_ratio < confidence_threshold:
        logger.warning(f"Low agreement between cameras ({agreement_ratio:.2f}). Using best guess.")

        if len(scores1) >= len(scores2):
            logger.info("Using Camera 1 results due to more detections")
            return scores1[:3]
        else:
            logger.info("Using Camera 2 results due to more detections")
            return scores2[:3]
        
    return validated_scores[:3]