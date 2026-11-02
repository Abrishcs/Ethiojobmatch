"""
EthioMatch — Learning Recommender
====================================
PURPOSE:
    For each missing skill identified in the gap analysis,
    suggest a specific learning resource (course, tutorial, or tool).

APPROACH:
    Hardcoded lookup table — simple, fast, reliable, works offline.
    Each skill maps to 1-2 curated resources that are:
      - Free or affordable
      - Reputable (university courses, official docs, top platforms)
      - Practical (not just theory)

    We can expand this table over time without changing any code logic.

WHY NOT DYNAMIC RECOMMENDATION?
    Dynamic approaches (API calls, web scraping) would add:
    - Internet dependency during inference
    - Latency for each query
    - Complexity with no clear quality gain
    For V1, curated resources are better than random search results.
"""


# ── Master resource table ─────────────────────────────────────────────────────
# Format: skill_name_lowercase → {name, url, type, cost}

LEARNING_RESOURCES: dict[str, list[dict]] = {
    # ── Programming Languages ──────────────────────────────────────────────
    "python": [
        {"name": "CS50's Introduction to Python (Harvard)", "url": "https://cs50.harvard.edu/python/", "type": "course", "cost": "Free"},
        {"name": "Automate the Boring Stuff with Python", "url": "https://automatetheboringstuff.com/", "type": "book", "cost": "Free"},
    ],
    "java": [
        {"name": "Java Programming MOOC (University of Helsinki)", "url": "https://java-programming.mooc.fi/", "type": "course", "cost": "Free"},
    ],
    "javascript": [
        {"name": "The Odin Project — JavaScript", "url": "https://www.theodinproject.com/paths/full-stack-javascript", "type": "course", "cost": "Free"},
        {"name": "JavaScript.info — Modern Tutorial", "url": "https://javascript.info/", "type": "tutorial", "cost": "Free"},
    ],
    "typescript": [
        {"name": "TypeScript Official Handbook", "url": "https://www.typescriptlang.org/docs/handbook/", "type": "docs", "cost": "Free"},
    ],
    "c++": [
        {"name": "LearnCpp.com", "url": "https://www.learncpp.com/", "type": "tutorial", "cost": "Free"},
    ],
    "c#": [
        {"name": "Microsoft C# Fundamentals", "url": "https://learn.microsoft.com/en-us/dotnet/csharp/", "type": "docs", "cost": "Free"},
    ],
    "go": [
        {"name": "Go by Example", "url": "https://gobyexample.com/", "type": "tutorial", "cost": "Free"},
    ],
    "kotlin": [
        {"name": "Kotlin Official — Learn Kotlin", "url": "https://kotlinlang.org/docs/getting-started.html", "type": "docs", "cost": "Free"},
    ],
    "r": [
        {"name": "R for Data Science (Hadley Wickham)", "url": "https://r4ds.hadley.nz/", "type": "book", "cost": "Free"},
    ],
    "sql": [
        {"name": "SQLBolt — Interactive SQL Lessons", "url": "https://sqlbolt.com/", "type": "tutorial", "cost": "Free"},
    ],

    # ── Web Frontend ───────────────────────────────────────────────────────
    "react": [
        {"name": "React Official Tutorial", "url": "https://react.dev/learn", "type": "docs", "cost": "Free"},
        {"name": "Full Stack Open (University of Helsinki)", "url": "https://fullstackopen.com/en/", "type": "course", "cost": "Free"},
    ],
    "angular": [
        {"name": "Angular Official Tutorial", "url": "https://angular.dev/tutorials", "type": "docs", "cost": "Free"},
    ],
    "vue": [
        {"name": "Vue.js Official Guide", "url": "https://vuejs.org/guide/introduction.html", "type": "docs", "cost": "Free"},
    ],
    "next.js": [
        {"name": "Next.js Learn Course", "url": "https://nextjs.org/learn", "type": "course", "cost": "Free"},
    ],
    "html": [
        {"name": "MDN Web Docs — HTML", "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML", "type": "docs", "cost": "Free"},
    ],
    "css": [
        {"name": "MDN Web Docs — CSS", "url": "https://developer.mozilla.org/en-US/docs/Learn/CSS", "type": "docs", "cost": "Free"},
    ],
    "tailwind css": [
        {"name": "Tailwind CSS Official Docs", "url": "https://tailwindcss.com/docs", "type": "docs", "cost": "Free"},
    ],

    # ── Web Backend ────────────────────────────────────────────────────────
    "node.js": [
        {"name": "The Odin Project — NodeJS", "url": "https://www.theodinproject.com/paths/full-stack-javascript/courses/nodejs", "type": "course", "cost": "Free"},
    ],
    "django": [
        {"name": "Django Official Tutorial", "url": "https://docs.djangoproject.com/en/stable/intro/tutorial01/", "type": "docs", "cost": "Free"},
        {"name": "Django for Beginners (William Vincent)", "url": "https://djangoforbeginners.com/", "type": "book", "cost": "Paid"},
    ],
    "flask": [
        {"name": "Flask Official Tutorial", "url": "https://flask.palletsprojects.com/en/stable/tutorial/", "type": "docs", "cost": "Free"},
    ],
    "fastapi": [
        {"name": "FastAPI Official Tutorial", "url": "https://fastapi.tiangolo.com/tutorial/", "type": "docs", "cost": "Free"},
    ],
    "spring boot": [
        {"name": "Spring Boot Official Guides", "url": "https://spring.io/guides", "type": "docs", "cost": "Free"},
    ],
    "rest api": [
        {"name": "RESTful API Design — Best Practices", "url": "https://restfulapi.net/", "type": "tutorial", "cost": "Free"},
    ],

    # ── Mobile ─────────────────────────────────────────────────────────────
    "flutter": [
        {"name": "Flutter Official Codelabs", "url": "https://docs.flutter.dev/get-started/codelab", "type": "course", "cost": "Free"},
    ],
    "react native": [
        {"name": "React Native Official Guide", "url": "https://reactnative.dev/docs/getting-started", "type": "docs", "cost": "Free"},
    ],

    # ── Databases ──────────────────────────────────────────────────────────
    "postgresql": [
        {"name": "PostgreSQL Tutorial", "url": "https://www.postgresqltutorial.com/", "type": "tutorial", "cost": "Free"},
    ],
    "mongodb": [
        {"name": "MongoDB University — Free Courses", "url": "https://learn.mongodb.com/", "type": "course", "cost": "Free"},
    ],
    "redis": [
        {"name": "Redis University", "url": "https://university.redis.com/", "type": "course", "cost": "Free"},
    ],
    "firebase": [
        {"name": "Firebase Official Docs", "url": "https://firebase.google.com/docs", "type": "docs", "cost": "Free"},
    ],

    # ── Cloud & DevOps ─────────────────────────────────────────────────────
    "docker": [
        {"name": "Docker Getting Started Guide", "url": "https://docs.docker.com/get-started/", "type": "docs", "cost": "Free"},
        {"name": "Docker Curriculum (Prakhar Srivastav)", "url": "https://docker-curriculum.com/", "type": "tutorial", "cost": "Free"},
    ],
    "kubernetes": [
        {"name": "Kubernetes Official Tutorial", "url": "https://kubernetes.io/docs/tutorials/", "type": "docs", "cost": "Free"},
    ],
    "aws": [
        {"name": "AWS Skill Builder — Free Courses", "url": "https://skillbuilder.aws/", "type": "course", "cost": "Free"},
    ],
    "azure": [
        {"name": "Microsoft Learn — Azure", "url": "https://learn.microsoft.com/en-us/training/azure/", "type": "course", "cost": "Free"},
    ],
    "google cloud": [
        {"name": "Google Cloud Skills Boost", "url": "https://www.cloudskillsboost.google/", "type": "course", "cost": "Free"},
    ],
    "ci/cd": [
        {"name": "GitHub Actions — Official Docs", "url": "https://docs.github.com/en/actions", "type": "docs", "cost": "Free"},
    ],
    "linux": [
        {"name": "Linux Journey", "url": "https://linuxjourney.com/", "type": "tutorial", "cost": "Free"},
    ],
    "git": [
        {"name": "Pro Git Book (Scott Chacon)", "url": "https://git-scm.com/book/en/v2", "type": "book", "cost": "Free"},
    ],
    "terraform": [
        {"name": "HashiCorp Terraform Tutorials", "url": "https://developer.hashicorp.com/terraform/tutorials", "type": "tutorial", "cost": "Free"},
    ],

    # ── Data & ML ──────────────────────────────────────────────────────────
    "machine learning": [
        {"name": "Andrew Ng — Machine Learning Specialization", "url": "https://www.coursera.org/specializations/machine-learning-introduction", "type": "course", "cost": "Free to audit"},
        {"name": "fast.ai — Practical ML", "url": "https://course.fast.ai/", "type": "course", "cost": "Free"},
    ],
    "deep learning": [
        {"name": "Andrew Ng — Deep Learning Specialization", "url": "https://www.coursera.org/specializations/deep-learning", "type": "course", "cost": "Free to audit"},
    ],
    "natural language processing": [
        {"name": "Hugging Face NLP Course", "url": "https://huggingface.co/learn/nlp-course", "type": "course", "cost": "Free"},
    ],
    "computer vision": [
        {"name": "CS231n — Stanford CV (YouTube)", "url": "https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv", "type": "course", "cost": "Free"},
    ],
    "data science": [
        {"name": "Kaggle Learn — Intro to Data Science", "url": "https://www.kaggle.com/learn", "type": "course", "cost": "Free"},
    ],
    "data analysis": [
        {"name": "Google Data Analytics Certificate", "url": "https://www.coursera.org/professional-certificates/google-data-analytics", "type": "course", "cost": "Free to audit"},
    ],
    "pandas": [
        {"name": "Kaggle — Pandas Course", "url": "https://www.kaggle.com/learn/pandas", "type": "course", "cost": "Free"},
    ],
    "tensorflow": [
        {"name": "TensorFlow Official Tutorials", "url": "https://www.tensorflow.org/tutorials", "type": "docs", "cost": "Free"},
    ],
    "pytorch": [
        {"name": "PyTorch Official Tutorials", "url": "https://pytorch.org/tutorials/", "type": "docs", "cost": "Free"},
    ],
    "scikit-learn": [
        {"name": "scikit-learn Official Tutorial", "url": "https://scikit-learn.org/stable/tutorial/", "type": "docs", "cost": "Free"},
    ],
    "power bi": [
        {"name": "Microsoft Learn — Power BI", "url": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi", "type": "course", "cost": "Free"},
    ],
    "tableau": [
        {"name": "Tableau Free Training Videos", "url": "https://www.tableau.com/learn/training", "type": "course", "cost": "Free"},
    ],

    # ── Security ───────────────────────────────────────────────────────────
    "cybersecurity": [
        {"name": "Cybersecurity for Beginners (Coursera/Google)", "url": "https://www.coursera.org/professional-certificates/google-cybersecurity", "type": "course", "cost": "Free to audit"},
    ],
    "network security": [
        {"name": "CompTIA Security+ Study Guide", "url": "https://www.comptia.org/certifications/security", "type": "certification", "cost": "Paid"},
    ],

    # ── Networking ─────────────────────────────────────────────────────────
    "cisco": [
        {"name": "Cisco Networking Academy — Free Courses", "url": "https://www.netacad.com/", "type": "course", "cost": "Free"},
    ],
    "ccna": [
        {"name": "Cisco CCNA Certification Prep", "url": "https://www.netacad.com/courses/networking/ccna-introduction-networks", "type": "course", "cost": "Free"},
    ],

    # ── Soft Skills ────────────────────────────────────────────────────────
    "communication": [
        {"name": "Improve Your English Communication Skills (Coursera)", "url": "https://www.coursera.org/specializations/improve-english", "type": "course", "cost": "Free to audit"},
    ],
    "project management": [
        {"name": "Google Project Management Certificate", "url": "https://www.coursera.org/professional-certificates/google-project-management", "type": "course", "cost": "Free to audit"},
    ],
    "agile": [
        {"name": "Agile with Atlassian Jira (Coursera)", "url": "https://www.coursera.org/learn/agile-atlassian-jira", "type": "course", "cost": "Free to audit"},
    ],

    # ── Tools ──────────────────────────────────────────────────────────────
    "autocad": [
        {"name": "AutoCAD Official Learning", "url": "https://www.autodesk.com/certification/learn", "type": "course", "cost": "Free"},
    ],
    "microservices": [
        {"name": "Microservices.io — Patterns", "url": "https://microservices.io/", "type": "tutorial", "cost": "Free"},
    ],
}


# ── Lookup function ────────────────────────────────────────────────────────────
def get_recommendations(missing_skills: list[str]) -> list[dict]:
    """
    For each missing skill, return learning resources.

    Args:
        missing_skills: List of skill names from skill gap analysis

    Returns:
        List of dicts:
        [
            {
                "skill": "Docker",
                "resources": [
                    {"name": "Docker Getting Started", "url": "...", ...},
                ]
            },
            ...
        ]

    Skills not in our lookup table get a generic suggestion.
    """
    recommendations = []

    for skill in missing_skills:
        skill_lower = skill.lower()
        resources = LEARNING_RESOURCES.get(skill_lower, [])

        if not resources:
            # Generic fallback: suggest searching on free platforms
            resources = [
                {
                    "name": f"Search '{skill}' on freeCodeCamp or Coursera",
                    "url": f"https://www.google.com/search?q={skill.replace(' ', '+')}+free+course",
                    "type": "search",
                    "cost": "Free",
                }
            ]

        recommendations.append({
            "skill": skill,
            "resources": resources,
        })

    return recommendations


# ── Script entry point (for testing) ──────────────────────────────────────────
if __name__ == "__main__":
    print("── Learning Recommender Test ──────────────────────────────")

    test_missing = ["Docker", "Kubernetes", "FastAPI", "CI/CD", "Redis", "Machine Learning"]
    recs = get_recommendations(test_missing)

    for rec in recs:
        print(f"\n❌ Missing: {rec['skill']}")
        for r in rec["resources"]:
            print(f"   📚 {r['name']}")
            print(f"      {r['url']} ({r['cost']})")

    print("\n✅ Recommender test complete")
