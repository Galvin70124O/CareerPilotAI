"""
Text cleaning shared by training AND the web app.

The ResumeAtlas dataset was already cleaned by its authors: lowercase, no punctuation
(it was deleted, so "asp.net" became "aspnet" and "node.js" became "nodejs").
User resumes are raw text, so we clean them the SAME way before the model sees them.
If training text and user text were cleaned differently, the model would not
recognise words like "aspnet" in a user's resume.
"""
import re

# These characters usually separate words ("python,java" or "sql;excel"), so they become spaces
SEPARATORS = r"[,;|()\[\]{}<>•●▪■◦·•]"


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(SEPARATORS, " ", text)
    # Delete every other punctuation mark, like the dataset did: "c++" -> "c", "ci/cd" -> "cicd"
    text = re.sub(r"[^\w\s]", "", text)
    # Remove leftover non-English/control characters and collapse whitespace
    text = re.sub(r"[^\x20-\x7e]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
