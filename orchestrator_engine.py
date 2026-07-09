"""
Mediora Orchestrator - Autonomous Hospital Workflow Intelligence
==================================================================

Same architectural pattern as sentinel_engine.py: this module never imports
Flask, SQLAlchemy, or `app`. It receives plain data (lists/dicts already
pulled from the database by app.py) and returns computed results. This keeps
all workflow math testable and free of hidden state.

All numbers here - current stage, delays, discharge readiness %, missed
follow-ups - come from real rows in CareJourneyEvent, DischargeChecklist,
and FollowUp. Nothing is invented. Gemini is only used at the very end, to
turn the already-computed numbers into readable executive sentences and
operational next-best-actions - never to decide a status, a diagnosis, or
a treatment.
"""

from datetime import datetime, date
import json
import re


# ---------------------------------------------------------------------------
# Fixed workflow definition - single source of truth, shared with app.py's
# seeding functions so the stage list is defined in exactly one place.
# ---------------------------------------------------------------------------

CARE_STAGES = [
    (1, 'Admission', 'General'),
    (2, 'Doctor Assignment', 'General'),
    (3, 'Investigations', 'Radiology / Laboratory'),
    (4, 'Treatment Started', 'General Medicine'),
    (5, 'Medication', 'Pharmacy'),
    (6, 'Observation', 'Nursing'),
    (7, 'Discharge Planning', 'Administration'),
    (8, 'Follow-up', 'Outpatient'),
]

DISCHARGE_TASKS = [
    'Doctor Approval',
    'Required Reports Ready',
    'Medication Prepared',
    'Billing Completed',
    'Discharge Summary Ready',
    'Follow-up Scheduled',
]

# How many hours a stage can sit "In Progress" before it's flagged as a
# bottleneck. These are operational assumptions, documented here so they can
# be tuned - not hidden inside a formula.
STAGE_DELAY_THRESHOLD_HOURS = {
    'Admission': 2,
    'Doctor Assignment': 4,
    'Investigations': 24,
    'Treatment Started': 48,
    'Medication': 12,
    'Observation': 24,
    'Discharge Planning': 24,
    'Follow-up': 0,  # follow-up delay is measured differently (missed date), not stage duration
}


def _hours_between(start, end):
    if not start:
        return None
    end = end or datetime.utcnow()
    return round((end - start).total_seconds() / 3600, 1)


# ---------------------------------------------------------------------------
# STEP 1: Per-patient Care Journey (capability #1 + #4)
# ---------------------------------------------------------------------------

def compute_patient_journey(events):
    """events: list of dicts, one per CareJourneyEvent row for one patient,
    each with: stage_name, stage_order, department, doctor_name, status,
    started_at, completed_at. Returns the full journey view."""

    events_sorted = sorted(events, key=lambda e: e['stage_order'])
    stages = []
    current_stage = None
    completed_count = 0

    for e in events_sorted:
        duration = _hours_between(e.get('started_at'), e.get('completed_at'))
        threshold = STAGE_DELAY_THRESHOLD_HOURS.get(e['stage_name'], 24)
        is_delayed = (
            e['status'] == 'In Progress' and
            e.get('started_at') is not None and
            _hours_between(e['started_at'], None) > threshold
        )
        stage_view = {
            'order': e['stage_order'],
            'name': e['stage_name'],
            'department': e['department'],
            'doctor_name': e.get('doctor_name'),
            'status': e['status'],
            'started_at': e.get('started_at').isoformat() if e.get('started_at') else None,
            'completed_at': e.get('completed_at').isoformat() if e.get('completed_at') else None,
            'duration_hours': duration,
            'is_delayed': bool(is_delayed),
        }
        stages.append(stage_view)
        if e['status'] == 'Completed':
            completed_count += 1
        if e['status'] == 'In Progress' and current_stage is None:
            current_stage = stage_view

    if current_stage is None:
        # nothing "In Progress" - either just started (all Pending) or fully done
        pending = [s for s in stages if s['status'] == 'Pending']
        current_stage = pending[0] if pending else (stages[-1] if stages else None)

    next_stage = None
    if current_stage:
        upcoming = [s for s in stages if s['order'] > current_stage['order']]
        next_stage = upcoming[0] if upcoming else None

    return {
        'stages': stages,
        'progress_pct': round((completed_count / len(stages)) * 100) if stages else 0,
        'current_stage': current_stage,
        'next_stage': next_stage,
        'current_department': current_stage['department'] if current_stage else None,
        'next_department': next_stage['department'] if next_stage else None,
    }


# ---------------------------------------------------------------------------
# STEP 2: Discharge Readiness (capability #5)
# ---------------------------------------------------------------------------

def compute_discharge_readiness(checklist_items):
    """checklist_items: list of dicts with task_name, is_complete."""
    total = len(checklist_items)
    if total == 0:
        return {'percent': 0, 'completed': [], 'remaining': DISCHARGE_TASKS[:]}

    completed = [c['task_name'] for c in checklist_items if c['is_complete']]
    remaining = [c['task_name'] for c in checklist_items if not c['is_complete']]
    return {
        'percent': round((len(completed) / total) * 100),
        'completed': completed,
        'remaining': remaining,
    }


# ---------------------------------------------------------------------------
# STEP 3: Follow-ups / Continuity of Care (capability #6)
# ---------------------------------------------------------------------------

def compute_followups(followups):
    """followups: list of dicts with scheduled_date (date), status, purpose.
    A 'Scheduled' follow-up whose date has already passed is treated as
    missed for display purposes even if no background job has updated its
    status yet - this is a real, derivable fact (today's date vs. the
    stored date), not a guess."""
    today = date.today()
    upcoming, missed, completed = [], [], []

    for f in followups:
        if f['status'] == 'Completed':
            completed.append(f)
        elif f['status'] == 'Missed':
            missed.append(f)
        elif f['status'] == 'Scheduled':
            if f['scheduled_date'] < today:
                missed.append({**f, 'status': 'Missed (overdue)'})
            else:
                upcoming.append(f)
        else:
            upcoming.append(f)  # Cancelled or unknown - shown but not flagged as urgent

    case_closed = len(followups) > 0 and len(upcoming) == 0 and len(missed) == 0

    return {
        'upcoming': upcoming,
        'missed': missed,
        'completed_count': len(completed),
        'case_closed': case_closed,
    }


# ---------------------------------------------------------------------------
# STEP 4: Hospital-wide aggregation - bottlenecks across all patients (#3)
# ---------------------------------------------------------------------------

def compute_hospital_bottlenecks(all_patient_journeys):
    """all_patient_journeys: list of {patient_name, journey} where journey is
    the output of compute_patient_journey(). Returns real, per-patient
    bottleneck entries - never a fabricated department-wide guess."""
    bottlenecks = []
    for entry in all_patient_journeys:
        stage = entry['journey']['current_stage']
        if stage and stage['is_delayed']:
            threshold = STAGE_DELAY_THRESHOLD_HOURS.get(stage['name'], 24)
            bottlenecks.append({
                'patient_name': entry['patient_name'],
                'stage': stage['name'],
                'department': stage['department'],
                'hours_in_stage': stage['duration_hours'],
                'threshold_hours': threshold,
            })
    return bottlenecks


# ---------------------------------------------------------------------------
# STEP 5: Build the Gemini prompt - operational only, evidence-enforced
# ---------------------------------------------------------------------------

def build_orchestrator_prompt(summary):
    """summary: a dict with hospital-wide counts and the real bottleneck/
    readiness/follow-up data computed above. Every instruction below forces
    the model to cite real figures and forbids clinical decision-making."""

    bottleneck_text = "None detected."
    if summary['bottlenecks']:
        bottleneck_text = "; ".join(
            f"{b['patient_name']}: stuck in {b['stage']} ({b['department']}) for "
            f"{b['hours_in_stage']}h, threshold is {b['threshold_hours']}h"
            for b in summary['bottlenecks'][:10]
        )

    low_readiness_text = "None."
    if summary['low_readiness_patients']:
        low_readiness_text = "; ".join(
            f"{p['patient_name']}: {p['percent']}% discharge-ready, remaining: {', '.join(p['remaining'])}"
            for p in summary['low_readiness_patients'][:10]
        )

    missed_followup_text = "None."
    if summary['missed_followups']:
        missed_followup_text = "; ".join(
            f"{f['patient_name']}: follow-up was due {f['scheduled_date']}"
            for f in summary['missed_followups'][:10]
        )

    prompt = f"""You are Mediora Orchestrator, an AI operations coordinator for a hospital. You are given ONLY real, already-computed operational data below. Do not invent any number, name, or fact not present here.

You provide OPERATIONAL guidance only - never diagnose a disease, never prescribe or suggest a medicine or treatment, never override a doctor's clinical judgement. Your job is coordinating workflow, not practicing medicine.

=== VERIFIED WORKFLOW DATA ===
Total admitted patients tracked: {summary['total_patients']}
Patients with an active workflow bottleneck: {len(summary['bottlenecks'])}
Bottleneck details: {bottleneck_text}
Patients with low discharge readiness (below 70%): {len(summary['low_readiness_patients'])}
Low-readiness details: {low_readiness_text}
Missed follow-ups: {len(summary['missed_followups'])}
Missed follow-up details: {missed_followup_text}

=== YOUR TASK ===
Return ONLY a valid JSON object, no markdown fences, no commentary, with EXACTLY this structure:

{{
  "executive_summary": ["<observation 1>", "... up to 5 total, one sentence each, each referencing a specific figure or name from the data above>"],
  "next_best_actions": [
    {{"action": "<concrete operational step, e.g. 'Schedule pending investigation', 'Notify pharmacy', 'Prepare discharge documentation'>", "priority": "<Low|Medium|High>", "reason": "<must cite the specific patient/figure from the data above>", "department": "<responsible department>", "expected_impact": "<short operational benefit>"}}
  ]
}}

Rules:
- Every executive_summary line and every next_best_actions entry MUST cite a real patient name, stage, department, percentage, or hour figure from the data above.
- Never invent a clinical recommendation (no medicines, no diagnoses, no treatment changes) - only operational/administrative actions (scheduling, notifying, documentation, coordination).
- If a category has no real signal, state that it is healthy rather than inventing a problem.
- At most 5 executive_summary items and 5 next_best_actions."""

    return prompt


def parse_orchestrator_ai_response(raw_text):
    cleaned = (raw_text or '').strip()
    cleaned = re.sub(r'^```json\s*|^```\s*|```\s*$', '', cleaned, flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not match:
            raise ValueError("AI returned an unreadable response.")
        data = json.loads(match.group(0))

    required = {'executive_summary', 'next_best_actions'}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"AI response missing fields: {', '.join(missing)}")
    return data


# ---------------------------------------------------------------------------
# STEP 6: Single entry point app.py calls for the hospital-wide view
# ---------------------------------------------------------------------------

def run_orchestrator_scan(patients_raw, genai_model_caller):
    """
    patients_raw: list of dicts, one per admitted patient, each with:
        'patient_name': str
        'journey_events': list of event dicts (see compute_patient_journey)
        'checklist_items': list of dicts (see compute_discharge_readiness)
        'followups': list of dicts (see compute_followups)

    genai_model_caller: callable(prompt_str) -> raw text response from Gemini.

    Returns everything the frontend needs in one JSON-ready dict.
    """
    patients_view = []
    all_journeys_for_bottlenecks = []
    low_readiness_patients = []
    missed_followups_all = []

    for p in patients_raw:
        journey = compute_patient_journey(p['journey_events'])
        readiness = compute_discharge_readiness(p['checklist_items'])
        followups = compute_followups(p['followups'])

        patients_view.append({
            'patient_name': p['patient_name'],
            'patient_id': p['patient_id'],
            'journey': journey,
            'readiness': readiness,
            'followups': followups,
        })

        all_journeys_for_bottlenecks.append({'patient_name': p['patient_name'], 'journey': journey})

        if readiness['percent'] < 70 and journey['current_stage'] and journey['current_stage']['name'] in ('Discharge Planning',):
            low_readiness_patients.append({
                'patient_name': p['patient_name'],
                'percent': readiness['percent'],
                'remaining': readiness['remaining'],
            })

        for f in followups['missed']:
            missed_followups_all.append({'patient_name': p['patient_name'], 'scheduled_date': f['scheduled_date'].isoformat()})

    bottlenecks = compute_hospital_bottlenecks(all_journeys_for_bottlenecks)

    summary = {
        'total_patients': len(patients_raw),
        'bottlenecks': bottlenecks,
        'low_readiness_patients': low_readiness_patients,
        'missed_followups': missed_followups_all,
    }

    ai_data = {'executive_summary': [], 'next_best_actions': []}
    try:
        prompt = build_orchestrator_prompt(summary)
        raw_text = genai_model_caller(prompt)
        ai_data = parse_orchestrator_ai_response(raw_text)
    except Exception as e:
        ai_data = {'executive_summary': [f'AI summary unavailable: {str(e)}'], 'next_best_actions': []}

    return {
        'patients': patients_view,
        'bottlenecks': bottlenecks,
        'executive_summary': ai_data.get('executive_summary', [])[:5],
        'next_best_actions': ai_data.get('next_best_actions', [])[:5],
        'scan_time': datetime.utcnow().isoformat() + 'Z',
    }