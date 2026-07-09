"""
Mediora Sentinel - Autonomous Hospital Intelligence Engine
============================================================

This module is intentionally decoupled from Flask and SQLAlchemy's app/db
objects. It receives already-bound model classes (Patient, Doctor, Bed,
Medicine, Attendance) from app.py and queries them directly via their own
`.query` interface. It never imports `app`, so there is no circular import
and the scoring/prompt logic can be unit-tested in isolation.

Sentinel is STATELESS by design: every call to run_sentinel_scan() re-derives
everything from the live database at that instant. Nothing is written to
storage. "Last Scan Time" is simply the UTC timestamp of the current
computation, not a persisted record.

------------------------------------------------------------------------
HOSPITAL PULSE SCORE - WEIGHTED MODEL (documented for transparency)
------------------------------------------------------------------------
The Pulse Score is a weighted sum of five sub-scores, each 0-100.
Weights were chosen to reflect operational impact, and always sum to 100
so the final score is itself 0-100:

    Bed Occupancy Health         25%
    Doctor Availability          20%
    Medicine Stock Health        20%
    Patient Care Continuity      20%
    Admission Volume Stability   15%
    ---------------------------------
    TOTAL                       100%

Each sub-score is computed by a plain, documented formula (see
`_score_*` functions below) - no black box, no LLM involved in the number
itself. The LLM is only used afterward, to narrate *why* the numbers look
the way they do and what to do about it, always grounded in these same
figures.
"""

from datetime import date, timedelta
import json
import re


# ---------------------------------------------------------------------------
# Tunable constants (kept in one place so thresholds are easy to justify/change)
# ---------------------------------------------------------------------------

LONG_STAY_THRESHOLD_DAYS = 7        # admitted longer than this -> "needs review" proxy
MEDICINE_EXPIRY_WINDOW_DAYS = 30    # expiring within this window counts as at-risk stock
WORKLOAD_IMBALANCE_RATIO = 1.5      # a doctor with > 1.5x their department's average patient
                                     # load is flagged as imbalanced

PULSE_WEIGHTS = {
    "bed_occupancy": 25,
    "doctor_availability": 20,
    "medicine_stock": 20,
    "patient_continuity": 20,
    "admission_volume": 15,
}


def _clamp(value, lo=0, hi=100):
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# STEP 1: Pull real metrics straight from the existing tables
# ---------------------------------------------------------------------------

def get_hospital_metrics(Patient, Doctor, Bed, Medicine, Attendance):
    """Query the existing models and return one flat dict of real numbers.
    No value here is invented - anything that can't be computed because a
    table is empty is explicitly marked so scoring/confidence can account
    for it honestly instead of pretending the data exists."""

    today = date.today()
    week_ago = today - timedelta(days=7)
    two_weeks_ago = today - timedelta(days=14)

    metrics = {}

    # --- Beds ---
    total_beds = Bed.query.count()
    occupied_beds = Bed.query.filter_by(status='Occupied').count()
    available_beds = Bed.query.filter_by(status='Available').count()
    maintenance_beds = Bed.query.filter_by(status='Maintenance').count()
    metrics['total_beds'] = total_beds
    metrics['occupied_beds'] = occupied_beds
    metrics['available_beds'] = available_beds
    metrics['maintenance_beds'] = maintenance_beds
    metrics['occupancy_rate'] = round((occupied_beds / total_beds) * 100, 1) if total_beds else None

    # --- Doctors & today's attendance ---
    active_doctors = Doctor.query.filter_by(status='Active').all()
    total_active_doctors = len(active_doctors)
    present_today = Attendance.query.filter_by(date=today, status='Present').count()
    absent_today = Attendance.query.filter_by(date=today, status='Absent').count()
    late_today = Attendance.query.filter_by(date=today, status='Late').count()
    metrics['total_active_doctors'] = total_active_doctors
    metrics['present_today'] = present_today
    metrics['absent_today'] = absent_today
    metrics['late_today'] = late_today
    metrics['availability_rate'] = (
        round((present_today / total_active_doctors) * 100, 1)
        if total_active_doctors else None
    )

    # --- Doctor workload balance (real per-doctor patient counts, grouped by department) ---
    department_loads = {}   # department -> list of (doctor_name, active_patient_count)
    for doc in active_doctors:
        patient_count = Patient.query.filter_by(doctor_id=doc.id, status='Admitted').count()
        department_loads.setdefault(doc.department, []).append((doc.name, patient_count))

    workload_imbalances = []
    for dept, loads in department_loads.items():
        if len(loads) < 2:
            continue  # need at least 2 doctors in a department for "average" to be meaningful
        avg_load = sum(c for _, c in loads) / len(loads)
        if avg_load <= 0:
            continue
        for name, count in loads:
            if count > avg_load * WORKLOAD_IMBALANCE_RATIO:
                workload_imbalances.append({
                    'doctor': name,
                    'department': dept,
                    'patient_count': count,
                    'department_average': round(avg_load, 1),
                })
    metrics['workload_imbalances'] = workload_imbalances

    # --- Medicines ---
    total_medicines = Medicine.query.count()
    low_stock_meds = Medicine.query.filter(Medicine.quantity <= Medicine.low_stock_threshold).all()
    expiry_cutoff = today + timedelta(days=MEDICINE_EXPIRY_WINDOW_DAYS)
    expiring_meds = Medicine.query.filter(
        Medicine.expiry_date != None,
        Medicine.expiry_date <= expiry_cutoff,
        Medicine.expiry_date >= today
    ).all()
    metrics['total_medicines'] = total_medicines
    metrics['low_stock_count'] = len(low_stock_meds)
    metrics['low_stock_items'] = [
        {'name': m.name, 'quantity': m.quantity, 'threshold': m.low_stock_threshold}
        for m in low_stock_meds
    ]
    metrics['expiring_count'] = len(expiring_meds)
    metrics['expiring_items'] = [
        {'name': m.name, 'expiry_date': m.expiry_date.isoformat()}
        for m in expiring_meds
    ]

    # --- Patients / continuity ---
    admitted_patients = Patient.query.filter_by(status='Admitted').all()
    total_admitted = len(admitted_patients)
    long_stay_patients = [
        p for p in admitted_patients
        if p.admission_date and (today - p.admission_date).days > LONG_STAY_THRESHOLD_DAYS
    ]
    metrics['total_admitted'] = total_admitted
    metrics['long_stay_count'] = len(long_stay_patients)
    metrics['long_stay_patients'] = [
        {'name': p.name, 'days_admitted': (today - p.admission_date).days}
        for p in long_stay_patients
    ]

    # --- Admission volume trend (real timestamps, no fabricated "emergency" flag) ---
    admissions_this_week = Patient.query.filter(
        Patient.admission_date >= week_ago, Patient.admission_date <= today
    ).count()
    admissions_prior_week = Patient.query.filter(
        Patient.admission_date >= two_weeks_ago, Patient.admission_date < week_ago
    ).count()
    metrics['admissions_this_week'] = admissions_this_week
    metrics['admissions_prior_week'] = admissions_prior_week
    if admissions_prior_week > 0:
        metrics['admission_change_pct'] = round(
            ((admissions_this_week - admissions_prior_week) / admissions_prior_week) * 100, 1
        )
    else:
        metrics['admission_change_pct'] = None  # not enough history to compute a trend honestly

    return metrics


# ---------------------------------------------------------------------------
# STEP 2: Sub-scores - each one documented, each one 0-100
# ---------------------------------------------------------------------------

def _score_bed_occupancy(m):
    """Occupancy is healthy up to ~70%, tolerable to 85%, risky beyond that.
    Returns (score, note) so the explanation can quote the exact rate."""
    rate = m['occupancy_rate']
    if rate is None:
        return 60, "no beds registered - neutral default score used"
    if rate <= 70:
        score = 100
    elif rate <= 85:
        score = 100 - (rate - 70) * 2          # 100 -> 70
    elif rate <= 95:
        score = 70 - (rate - 85) * 4            # 70 -> 30
    else:
        score = 30 - (rate - 95) * 6            # 30 -> 0 at 100%
    return _clamp(round(score, 1)), f"{rate}% bed occupancy ({m['occupied_beds']}/{m['total_beds']} beds)"


def _score_doctor_availability(m):
    rate = m['availability_rate']
    if rate is None:
        return 60, "no active doctors registered - neutral default score used"
    score = _clamp(round(rate, 1))
    note = f"{rate}% of active doctors present today ({m['present_today']}/{m['total_active_doctors']})"
    return score, note


def _score_medicine_stock(m):
    total = m['total_medicines']
    if total == 0:
        return 60, "no medicines registered - neutral default score used"
    low_ratio = m['low_stock_count'] / total
    expiring_ratio = m['expiring_count'] / total
    score = 100 - (low_ratio * 70) - (expiring_ratio * 30)
    note = (f"{m['low_stock_count']}/{total} items at/below reorder threshold, "
            f"{m['expiring_count']}/{total} expiring within {MEDICINE_EXPIRY_WINDOW_DAYS} days")
    return _clamp(round(score, 1)), note


def _score_patient_continuity(m):
    total = m['total_admitted']
    if total == 0:
        return 100, "no currently admitted patients"
    long_stay_ratio = m['long_stay_count'] / total
    score = 100 - (long_stay_ratio * 100)
    note = (f"{m['long_stay_count']}/{total} admitted patients have exceeded "
            f"{LONG_STAY_THRESHOLD_DAYS} days without discharge")
    return _clamp(round(score, 1)), note


def _score_admission_volume(m):
    change = m['admission_change_pct']
    if change is None:
        return 75, f"only {m['admissions_this_week']} admissions in the last 7 days - insufficient history for a trend"
    if change <= 20:
        score = 100
    else:
        score = 100 - (change - 20) * 1.5
    note = (f"{m['admissions_this_week']} admissions this week vs {m['admissions_prior_week']} "
            f"the previous week ({change:+.1f}%)")
    return _clamp(round(score, 1), lo=20), note


def compute_pulse_score(metrics):
    """Returns a dict with the final weighted score, per-component breakdown
    (each with its weight, sub-score and the real data behind it), a status
    label, a confidence rating based on data completeness, and a plain-text
    explanation - all deterministic, all traceable back to compute_pulse_score's
    inputs, with no LLM involved."""

    components = {
        'bed_occupancy': _score_bed_occupancy(metrics),
        'doctor_availability': _score_doctor_availability(metrics),
        'medicine_stock': _score_medicine_stock(metrics),
        'patient_continuity': _score_patient_continuity(metrics),
        'admission_volume': _score_admission_volume(metrics),
    }

    weighted_total = 0.0
    breakdown = []
    data_backed_count = 0
    for key, (score, note) in components.items():
        weight = PULSE_WEIGHTS[key]
        contribution = round((score / 100) * weight, 1)
        weighted_total += contribution
        is_data_backed = "neutral default" not in note and "insufficient history" not in note
        if is_data_backed:
            data_backed_count += 1
        breakdown.append({
            'metric': key,
            'weight_pct': weight,
            'sub_score': score,
            'points_contributed': contribution,
            'evidence': note,
        })

    final_score = _clamp(round(weighted_total))

    if final_score >= 85:
        status = "Excellent"
    elif final_score >= 70:
        status = "Stable"
    elif final_score >= 50:
        status = "Needs Attention"
    else:
        status = "Critical"

    confidence_pct = round((data_backed_count / len(components)) * 100)

    explanation_lines = [f"{b['metric'].replace('_', ' ').title()}: {b['points_contributed']}/{b['weight_pct']} pts "
                          f"({b['evidence']})" for b in breakdown]
    explanation = "Pulse Score of " + str(final_score) + "/100 (" + status + ") is the weighted sum of: " + \
        "; ".join(explanation_lines) + "."

    return {
        'score': final_score,
        'status': status,
        'confidence_pct': confidence_pct,
        'breakdown': breakdown,
        'explanation': explanation,
    }


# ---------------------------------------------------------------------------
# STEP 3: Build the Gemini prompt - every instruction forces evidence citation
# ---------------------------------------------------------------------------

def build_sentinel_prompt(metrics, pulse):
    """Builds a prompt that hands the LLM only real, already-computed numbers
    and explicitly forbids inventing figures. Every recommendation/risk must
    cite one of the provided data points."""

    workload_text = "None detected."
    if metrics['workload_imbalances']:
        workload_text = "; ".join(
            f"Dr. {w['doctor']} ({w['department']}) has {w['patient_count']} active patients vs a "
            f"department average of {w['department_average']}"
            for w in metrics['workload_imbalances']
        )

    low_stock_text = "None."
    if metrics['low_stock_items']:
        low_stock_text = "; ".join(
            f"{item['name']}: {item['quantity']} units left (reorder threshold {item['threshold']})"
            for item in metrics['low_stock_items'][:10]
        )

    expiring_text = "None."
    if metrics['expiring_items']:
        expiring_text = "; ".join(
            f"{item['name']} expires {item['expiry_date']}" for item in metrics['expiring_items'][:10]
        )

    long_stay_text = "None."
    if metrics['long_stay_patients']:
        long_stay_text = "; ".join(
            f"{p['name']} ({p['days_admitted']} days admitted)" for p in metrics['long_stay_patients'][:10]
        )

    prompt = f"""You are Mediora Sentinel, an AI Chief Operations Officer for a hospital. You are given ONLY real, already-verified operational data below. Do not invent any number, name, or fact that is not present in this data.

=== VERIFIED HOSPITAL DATA ===
Bed occupancy: {metrics['occupancy_rate']}% ({metrics['occupied_beds']}/{metrics['total_beds']} occupied, {metrics['available_beds']} available, {metrics['maintenance_beds']} under maintenance)
Doctor availability today: {metrics['availability_rate']}% ({metrics['present_today']}/{metrics['total_active_doctors']} active doctors present, {metrics['absent_today']} absent, {metrics['late_today']} late)
Doctor workload imbalances: {workload_text}
Medicine inventory: {metrics['total_medicines']} items tracked, {metrics['low_stock_count']} at/below reorder threshold, {metrics['expiring_count']} expiring within {MEDICINE_EXPIRY_WINDOW_DAYS} days
Low-stock items: {low_stock_text}
Expiring items: {expiring_text}
Admitted patients: {metrics['total_admitted']} total, {metrics['long_stay_count']} exceeding {LONG_STAY_THRESHOLD_DAYS} days without discharge
Long-stay patients (candidates for clinical review): {long_stay_text}
Admission volume: {metrics['admissions_this_week']} this week vs {metrics['admissions_prior_week']} last week (change: {metrics['admission_change_pct']}%)
Hospital Pulse Score: {pulse['score']}/100 ({pulse['status']}, confidence {pulse['confidence_pct']}%)

=== YOUR TASK ===
Return ONLY a valid JSON object, no markdown fences, no commentary, with EXACTLY this structure:

{{
  "mission_brief": ["<observation 1>", "... up to 5 total, each one sentence, each referencing a specific number from the data above>"],
  "risks": [
    {{"title": "<short risk name>", "level": "<Low|Medium|High>", "confidence_pct": <0-100 integer>, "reason": "<must cite the specific data point that triggered this risk>", "recommended_action": "<concrete next step>"}}
  ],
  "recommendations": [
    {{"action": "<what should be done, concrete and specific>", "reason": "<why - must cite the exact figure(s) from the data above, e.g. 'X units left vs threshold of Y', or 'Z% occupancy'>"}}
  ]
}}

Rules:
- Every single mission_brief line, risk, and recommendation MUST explicitly reference a real figure from the data above (a percentage, a count, a name, a comparison to a threshold or average). A sentence with no cited number is not acceptable.
- If a category has no real signal (e.g. no low-stock items), do not invent a risk for it - simply omit it or state that it is healthy.
- Generate at most 5 mission_brief items, at most 5 risks, at most 5 recommendations.
- Do not mention patient medical diagnoses or private clinical details beyond what's given (names/day-counts only, for operational review purposes)."""

    return prompt


# ---------------------------------------------------------------------------
# STEP 4: Robust JSON parsing (same defensive pattern as the AI Twin route)
# ---------------------------------------------------------------------------

def parse_sentinel_ai_response(raw_text):
    """Strips markdown fences if present and parses JSON, falling back to
    extracting the first {...} block if the model added stray text.
    Raises ValueError with a clear message on failure so the route can
    return a clean error response instead of a stack trace."""

    cleaned = (raw_text or '').strip()
    cleaned = re.sub(r'^```json\s*|^```\s*|```\s*$', '', cleaned, flags=re.MULTILINE).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not match:
            raise ValueError("AI returned an unreadable response.")
        data = json.loads(match.group(0))

    required_keys = {'mission_brief', 'risks', 'recommendations'}
    missing = required_keys - data.keys()
    if missing:
        raise ValueError(f"AI response missing fields: {', '.join(missing)}")

    return data


# ---------------------------------------------------------------------------
# STEP 5: One entry point app.py calls - ties everything together, stateless
# ---------------------------------------------------------------------------

def run_sentinel_scan(Patient, Doctor, Bed, Medicine, Attendance, genai_model_caller):
    """
    genai_model_caller: a callable that takes a prompt string and returns the
    raw text response from Gemini. Passed in from app.py so this module never
    imports google.generativeai or touches API keys directly.

    Returns a single dict ready to be jsonified straight to the frontend.
    Nothing is persisted.
    """
    from datetime import datetime

    metrics = get_hospital_metrics(Patient, Doctor, Bed, Medicine, Attendance)
    pulse = compute_pulse_score(metrics)
    prompt = build_sentinel_prompt(metrics, pulse)
    raw_text = genai_model_caller(prompt)
    ai_data = parse_sentinel_ai_response(raw_text)

    return {
        'pulse': pulse,
        'metrics': metrics,
        'mission_brief': ai_data.get('mission_brief', [])[:5],
        'risks': ai_data.get('risks', [])[:5],
        'recommendations': ai_data.get('recommendations', [])[:5],
        'scan_time': datetime.utcnow().isoformat() + 'Z',
    }