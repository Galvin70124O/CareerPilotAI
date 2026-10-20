"""
Names shown to the user for each career.

The model and the data keep the original category names (e.g. "Advocate"). Some names are
misleading on their own, so we show a clearer name on the page. Add more here if needed.
"""
DISPLAY_NAMES = {
    # The "Advocate" resumes are lawyers AND victim/patient/customer advocates
    "Advocate": "Advocate (Legal & Social Services)",
}


def display_name(career):
    return DISPLAY_NAMES.get(career, career)
