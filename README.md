# Tech Stack Recommender

A content-based recommendation system that suggests tech career roles from the skills you know. Built as **Project 3** of the DecodeLabs Artificial Intelligence Industrial Training Kit (Batch 2026).

Enter at least three skills and the program ranks 29 job roles by how closely they match your profile. It shows the Top 3, together with the skills you already have and the skills you still need to learn.

## How it works

The project follows the **Input, Process, Output** model:

| Stage | What happens |
|---|---|
| Input | The user types 3 or more skills |
| Process | Skills become **TF-IDF** vectors, then **cosine similarity** scores every role in `raw_skills.csv` |
| Output | The Top 3 roles, sorted by score |

The 4-step ranking pipeline from the course:

1. **Ingestion**: read and clean the user's skills
2. **Scoring**: cosine similarity against every job role
3. **Sorting**: highest score first
4. **Filtering**: keep only the Top 3

Why these techniques:

- **TF-IDF** gives specific skills (like `kubernetes`) more weight than skills found in many roles (like `python`).
- **Cosine similarity** compares the direction of two skill vectors, not their size.
- **Content-based filtering** works from the roles' own attributes, so no history of other users is needed.

## Example

```
Enter at least 3 skills separated by commas (or type 'quit'): docker, kubernetes, linux

Your skills: docker, kubernetes, linux

=== Top 3 career matches ===

1. DevOps Engineer - 53% match
   You already have : docker, kubernetes, linux
   Skills to learn  : ci/cd, aws, automation, terraform, monitoring, git, cloud computing, python

2. MLOps Engineer - 47% match
   You already have : docker, kubernetes
   Skills to learn  : python, ci/cd, mlops, machine learning, cloud computing, git

3. Site Reliability Engineer - 42% match
   You already have : linux, kubernetes
   Skills to learn  : monitoring, automation, python, cloud computing, networking, ci/cd
```

## Getting started

Requires **Python 3.9+**.

```bash
git clone https://github.com/imran5775/Tech-Stack-Recommender.git
cd Tech-Stack-Recommender
pip install -r requirements.txt
python tech_stack_recommender.py
```

## Features

- Requires at least 3 skills, as the project specification asks
- Fixes small typos (`pyhton` becomes `python`) and understands short names (`ml`, `js`, `k8s`)
- Skips unknown skills and tells the user which ones were skipped
- Handles the **cold-start problem** with an onboarding step: it lists all available skills and keeps asking until enough valid skills are given
- Runs in a loop so you can try different skill sets

## Dataset

`raw_skills.csv` contains 29 roles and 68 skills across data, AI, cloud, DevOps, networking, security, web, mobile and design. It is a **sample dataset created for this project** from common industry job requirements, so it simplifies real job descriptions.

Add your own role by adding a line to the CSV and separating skills with semicolons:

```
Robotics Engineer,python;c++;linux;computer vision;algorithms
```

## Project structure

```
Tech-Stack-Recommender/
├── tech_stack_recommender.py   # the recommender
├── raw_skills.csv              # job roles and their skills
├── requirements.txt
├── README.md
└── .gitignore
```

## Future improvements

- Let users rate how well they know each skill (beginner to expert)
- Add learning resources for each missing skill
- Use a larger real-world job dataset
- Build a simple web interface

## Author

**Muhammad Imran**
DecodeLabs AI Internship, Batch 2026