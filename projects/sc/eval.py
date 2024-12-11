import re
from typing import List, Optional


def _regex_function(text: str, pattern: str, group_select: Optional[int] = None) -> Optional[str]:
    match = re.search(pattern, text)
    if match:
        if group_select is not None:
            return match.group(group_select)
        return match.group(0)
    return None


def _take_first(matches: List[Optional[str]]) -> Optional[str]:
    for match in matches:
        if match:
            return match
    return None


def strict_match(text: str) -> str:
    # Define the pattern for strict-match
    pattern = r"#### (\-?[0-9\.\,]+)"

    # Apply regex function
    match = _regex_function(text, pattern)

    # Apply take_first on the match
    first = _take_first([match])
    if not first:
        return ""
    return first


def flexible_extract(text: str) -> Optional[str]:
    # Define the pattern for flexible-extract
    pattern = r"(-?[$0-9\.,]{2,})|(-?[0-9]+)"

    # Apply regex function
    match = _regex_function(text, pattern, group_select=-1)

    # Apply take_first on the match
    return _take_first([match])
