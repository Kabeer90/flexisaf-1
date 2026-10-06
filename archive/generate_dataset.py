"""
Synthetic Student Placement Time-Series Dataset Generator
-----------------------------------------------------------
Generates a monthly-tracked, panel-style dataset simulating engineering
students' academic/skill trajectories leading up to campus placement,
suitable for LSTM-based sequence classification.

Every student has the SAME number of monthly records (fixed sequence
length), which makes it trivial to reshape into a (students, months,
features) tensor for an LSTM.
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEED = 42
N_STUDENTS = 1000
N_MONTHS = 18                      # fixed-length monthly sequence per student
START_DATE = "2023-01-01"          # tracking window start (5th semester onward)

rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# Per-student latent traits (drive the whole trajectory)
# ------------------------------------------------------------------
baseline_cgpa   = np.clip(rng.normal(7.0, 0.8, N_STUDENTS), 5.0, 9.6)
cgpa_growth     = rng.uniform(0.01, 0.05, N_STUDENTS)
apt_baseline    = np.clip(rng.normal(58, 15, N_STUDENTS), 25, 95)
soft_baseline   = np.clip(rng.normal(3.2, 0.7, N_STUDENTS), 1.0, 5.0)
diligence       = rng.uniform(0.25, 1.0, N_STUDENTS)          # drives skill-building pace
attend_baseline = np.clip(rng.normal(82, 10, N_STUDENTS), 45, 100)
initial_backlog = rng.choice([0, 0, 0, 1, 1, 2, 3], size=N_STUDENTS,
                              p=[0.45, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05])

dates = pd.date_range(START_DATE, periods=N_MONTHS, freq="MS")

rows = []

for sid in range(1, N_STUDENTS + 1):
    idx = sid - 1
    placed = False
    placed_month = None

    internships = 0
    projects = 0
    certs = 0
    backlog = initial_backlog[idx]
    mock_interviews = 0

    # small chance internships/projects/certs already >0 at window start
    internships = rng.choice([0, 0, 1], p=[0.7, 0.15, 0.15])
    projects = rng.choice([0, 1, 1, 2], p=[0.35, 0.35, 0.2, 0.1])
    certs = rng.choice([0, 0, 1], p=[0.6, 0.2, 0.2])

    for m in range(N_MONTHS):
        # ---- evolving features ----
        cgpa = np.clip(
            baseline_cgpa[idx] + cgpa_growth[idx] * m + rng.normal(0, 0.12),
            4.0, 10.0
        )
        attendance = np.clip(
            attend_baseline[idx] + rng.normal(0, 4), 35, 100
        )
        aptitude = np.clip(
            apt_baseline[idx] + diligence[idx] * m * 0.9 + rng.normal(0, 4),
            0, 100
        )
        soft_skills = np.clip(
            soft_baseline[idx] + 0.01 * m + rng.normal(0, 0.15), 1.0, 5.0
        )
        coding_hours = np.clip(
            diligence[idx] * 12 + rng.normal(0, 2.5), 0, 40
        )

        # cumulative counters increase with some monthly probability
        if rng.random() < 0.10 * diligence[idx]:
            internships += 1
        if rng.random() < 0.14 * diligence[idx]:
            projects += 1
        if rng.random() < 0.10 * diligence[idx]:
            certs += 1
        # mock interviews ramp up as placement season approaches
        season_factor = m / N_MONTHS
        if rng.random() < (0.05 + 0.35 * season_factor) * diligence[idx]:
            mock_interviews += 1
        # backlogs get cleared over time
        if backlog > 0 and rng.random() < 0.12:
            backlog -= 1

        # ---- latent readiness score ----
        readiness = (
            0.28 * (cgpa / 10) +
            0.16 * (aptitude / 100) +
            0.10 * (soft_skills / 5) +
            0.14 * min(internships, 3) / 3 +
            0.12 * min(projects, 5) / 5 +
            0.08 * min(certs, 3) / 3 +
            0.06 * min(coding_hours, 20) / 20 +
            0.10 * min(mock_interviews, 10) / 10 -
            0.18 * min(backlog, 3) / 3
        )
        noise = rng.normal(0, 0.05)
        score = readiness + noise

        # ---- monthly placement hazard (only if not already placed) ----
        # Hiring happens in waves (placement drives), not uniformly across
        # the year -- three drive peaks give the LSTM a real temporal
        # pattern to learn beyond the slow trend features.
        drive_centers = (6, 11, 16)
        seasonal_boost = max(
            np.exp(-0.5 * ((m - c) / 1.4) ** 2) for c in drive_centers
        )

        if not placed:
            z = 14.0 * (score - 0.61) + 1.0 * seasonal_boost
            prob = (1 / (1 + np.exp(-z))) * 0.23   # cap monthly hazard
            if rng.random() < prob:
                placed = True
                placed_month = m

        placement_status = 1 if placed else 0

        rows.append((
            dates[m].strftime("%Y-%m-%d"),
            sid,
            m + 1,
            round(float(cgpa), 2),
            round(float(attendance), 1),
            round(float(aptitude), 1),
            round(float(soft_skills), 2),
            int(internships),
            int(projects),
            int(certs),
            round(float(coding_hours), 1),
            int(mock_interviews),
            int(backlog),
            placement_status,
        ))

df = pd.DataFrame(rows, columns=[
    "date", "student_id", "month_index", "cgpa", "attendance_percentage",
    "aptitude_score", "soft_skills_rating", "internships_count",
    "projects_count", "certifications_count", "coding_practice_hours",
    "mock_interviews_attended", "active_backlogs", "placement_status",
])

out_path = "/home/claude/dataset_gen/placement_timeseries_synthetic.csv"
df.to_csv(out_path, index=False)

# ------------------------------------------------------------------
# Quick sanity report
# ------------------------------------------------------------------
final_month = df[df.month_index == N_MONTHS]
overall_rate = final_month.placement_status.mean()
print(f"Rows: {len(df)}  |  Students: {N_STUDENTS}  |  Months/student: {N_MONTHS}")
print(f"Final-month placement rate: {overall_rate:.1%}")
print(df.head(6).to_string(index=False))
print("...")
print(df[df.student_id == 1].to_string(index=False))
