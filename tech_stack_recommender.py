"""Tech Stack Recommender (DecodeLabs AI Internship, Project 3)
Author: Muhammad Imran

Idea: you type your skills, the program finds the job roles that match them best.

The program has 3 parts (Input -> Process -> Output):
    Input   : the user types at least 3 skills
    Process : skills are turned into numbers (TF-IDF) and compared
              with every job role using cosine similarity
    Output  : the Top 3 matching roles
"""

import csv
import textwrap
from difflib import get_close_matches
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Settings (change these if you like)
# ---------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "raw_skills.csv"  # file in the same folder
MIN_SKILLS = 3  # the user must enter at least this many skills
TOP_N = 3       # how many roles to show

# Short names people often type -> the real skill name in our dataset
ALIASES = {
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "natural language processing": "nlp",
    "cloud": "cloud computing",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "cicd": "ci/cd",
    "tf": "tensorflow",
    "bash": "bash scripting",
    "shell": "bash scripting",
    "api": "apis",
    "db": "databases",
    "database": "databases",
    "cybersecurity": "security",
    "pentesting": "ethical hacking",
    "test": "testing",
    "cpp": "c++",
    "csharp": "c#",
    "ui": "ui design",
    "ux": "ui design",
    "powerbi": "power bi",
}


# ---------------------------------------------------------------
# Small helper: "Python, SQL" -> ["python", "sql"]
# ---------------------------------------------------------------
def split_skills(text):
    """Split a comma separated text into a clean list of lowercase skills."""
    return [skill.strip().lower() for skill in text.split(",") if skill.strip()]


# ---------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------
def load_roles():
    """Read raw_skills.csv and return a list of roles.

    Each role looks like: {"name": "DevOps Engineer", "skills": "docker, aws, ..."}
    """
    roles = []
    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            skills = row["skills"].replace(";", ",")  # "a;b;c" -> "a,b,c"
            roles.append({"name": row["role"], "skills": skills})
    return roles


# ---------------------------------------------------------------
# Step 2: Turn every role into numbers (TF-IDF vectors)
# ---------------------------------------------------------------
def build_model(roles):
    """Learn the skill vocabulary and create one vector for every role.

    TF-IDF gives a high score to rare, specific skills (like kubernetes)
    and a low score to skills that appear in many roles (like python).
    """
    vectorizer = TfidfVectorizer(tokenizer=split_skills, token_pattern=None)
    role_vectors = vectorizer.fit_transform([role["skills"] for role in roles])
    return vectorizer, role_vectors


# ---------------------------------------------------------------
# Step 3: Clean what the user typed
# ---------------------------------------------------------------
def clean_user_skills(text, all_skills):
    """Return (valid skills, skipped skills) from the user's text."""
    valid = []
    skipped = []

    for skill in split_skills(text):
        skill = ALIASES.get(skill, skill)  # "ml" -> "machine learning"

        if skill not in all_skills:
            # Try to fix small spelling mistakes ("pyhton" -> "python")
            match = get_close_matches(skill, all_skills, n=1, cutoff=0.8)
            if match:
                print(f"  Interpreted '{skill}' as '{match[0]}'")
                skill = match[0]
            else:
                skipped.append(skill)
                continue

        if skill not in valid:  # ignore repeated skills
            valid.append(skill)

    return valid, skipped


# ---------------------------------------------------------------
# Step 4: Score, sort and filter (the recommendation engine)
# ---------------------------------------------------------------
def recommend(user_skills, roles, vectorizer, role_vectors):
    """Return the Top N roles that match the user's skills."""
    # 1. Turn the user's skills into a vector (same vocabulary as the roles)
    user_vector = vectorizer.transform([", ".join(user_skills)])

    # 2. Scoring: cosine similarity between the user and every role
    scores = cosine_similarity(user_vector, role_vectors)[0]

    # 3. Collect every role that has some match
    results = []
    for role, score in zip(roles, scores):
        if score > 0:
            role_skills = split_skills(role["skills"])
            have = [s for s in role_skills if s in user_skills]
            learn = [s for s in role_skills if s not in user_skills]
            results.append((score, role["name"], have, learn))

    # 4. Sorting: best score first, then Filtering: keep only the Top N
    results.sort(reverse=True)
    return results[:TOP_N]


# ---------------------------------------------------------------
# Step 5: Show the result
# ---------------------------------------------------------------
def show_results(results):
    print(f"\n=== Top {len(results)} career matches ===")
    for number, (score, name, have, learn) in enumerate(results, start=1):
        print(f"\n{number}. {name} - {score:.0%} match")
        print(f"   You already have : {', '.join(have)}")
        if learn:
            print(f"   Skills to learn  : {', '.join(learn)}")


# ---------------------------------------------------------------
# Main program
# ---------------------------------------------------------------
def main():
    roles = load_roles()
    vectorizer, role_vectors = build_model(roles)
    all_skills = sorted(vectorizer.vocabulary_)  # every skill in the dataset

    print("=" * 55)
    print("  Tech Stack Recommender | by Muhammad Imran")
    print("=" * 55)
    print(f"Loaded {len(roles)} job roles and {len(all_skills)} skills.\n")
    print("Available skills:")
    print(textwrap.fill(", ".join(all_skills), width=70, initial_indent="  ",
                        subsequent_indent="  "))

    while True:
        text = input(f"\nEnter at least {MIN_SKILLS} skills separated by commas "
                     "(or type 'quit'): ")

        if text.strip().lower() in ("quit", "exit"):
            break

        user_skills, skipped = clean_user_skills(text, all_skills)

        for skill in skipped:
            print(f"  Skipped '{skill}' (not in the skill list)")

        # Cold start: with too few skills we cannot match well, so ask again
        if len(user_skills) < MIN_SKILLS:
            print(f"Please enter at least {MIN_SKILLS} valid skills from the list above.")
            continue

        print("\nYour skills:", ", ".join(user_skills))
        show_results(recommend(user_skills, roles, vectorizer, role_vectors))

        again = input("\nTry again with different skills? (y/n): ").strip().lower()
        if again != "y":
            break

    print("\nGoodbye! Good luck with your career.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")