import os
import json
import re
import calendar
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from sqlalchemy import extract, func

import sentinel_engine
import orchestrator_engine
from orchestrator_engine import CARE_STAGES, DISCHARGE_TASKS

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mediora-secret-key-2024')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access Mediora.'
login_manager.login_message_category = 'warning'

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class Admin(db.Model):
    __tablename__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    hospital_name = db.Column(db.String(200), default='Mediora Health Centre')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def is_authenticated(self): return True
    @property
    def is_active(self): return True
    @property
    def is_anonymous(self): return False
    def get_id(self): return str(self.id)
    def set_password(self, p): self.password_hash = generate_password_hash(p)
    def check_password(self, p): return check_password_hash(self.password_hash, p)

class Doctor(db.Model):
    __tablename__ = 'doctors'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    phone = db.Column(db.String(15))
    email = db.Column(db.String(120))
    experience_years = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='Active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('Patient', backref='doctor', lazy=True)
    attendance_records = db.relationship('Attendance', backref='doctor', lazy=True)

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    phone = db.Column(db.String(15))
    address = db.Column(db.Text)
    disease = db.Column(db.String(200))
    medical_history = db.Column(db.Text)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'))
    bed_id = db.Column(db.Integer, db.ForeignKey('beds.id'))
    admission_date = db.Column(db.Date, default=date.today)
    discharge_date = db.Column(db.Date)
    status = db.Column(db.String(20), default='Admitted')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Medicine(db.Model):
    __tablename__ = 'medicines'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    category = db.Column(db.String(100))
    manufacturer = db.Column(db.String(150))
    quantity = db.Column(db.Integer, default=0)
    unit = db.Column(db.String(30), default='Tablets')
    price_per_unit = db.Column(db.Float, default=0.0)
    low_stock_threshold = db.Column(db.Integer, default=50)
    expiry_date = db.Column(db.Date)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def is_low_stock(self): return self.quantity <= self.low_stock_threshold
    @property
    def is_expired(self): return self.expiry_date < date.today() if self.expiry_date else False
    @property
    def status(self):
        if self.is_expired: return 'Expired'
        if self.quantity == 0: return 'Out of Stock'
        if self.is_low_stock: return 'Low Stock'
        return 'Available'

class Bed(db.Model):
    __tablename__ = 'beds'
    id = db.Column(db.Integer, primary_key=True)
    bed_number = db.Column(db.String(20), unique=True, nullable=False)
    ward = db.Column(db.String(100), nullable=False)
    bed_type = db.Column(db.String(50), default='General')
    status = db.Column(db.String(20), default='Available')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patients = db.relationship('Patient', backref='bed', lazy=True)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    date = db.Column(db.Date, default=date.today)
    status = db.Column(db.String(20), default='Present')
    check_in_time = db.Column(db.String(10))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ActivityLog(db.Model):
    __tablename__ = 'activity_log'
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50))
    icon = db.Column(db.String(50), default='bi-activity')
    color = db.Column(db.String(20), default='primary')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class HealthTwinAnalysis(db.Model):
    """Stores each AI Health Twin analysis generated for a patient, so the
    twin has persistent memory across sessions and can reason about trends
    (improving/declining) instead of generating a stateless snapshot each time."""
    __tablename__ = 'health_twin_analysis'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    overall_score = db.Column(db.Integer)
    overall_status = db.Column(db.String(50))
    analysis_json = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    patient = db.relationship('Patient', backref=db.backref(
        'health_twin_analyses', lazy=True, order_by='HealthTwinAnalysis.created_at.desc()'))

class CareJourneyEvent(db.Model):
    """One row per patient per care stage (8 fixed stages, see CARE_STAGES
    in orchestrator_engine.py). This is the real, staff-updated record of
    where a patient actually is - nothing about the journey is guessed."""
    __tablename__ = 'care_journey_events'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    stage_name = db.Column(db.String(50), nullable=False)
    stage_order = db.Column(db.Integer, nullable=False)
    department = db.Column(db.String(100))
    assigned_doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=True)
    status = db.Column(db.String(20), default='Pending')  # Pending / In Progress / Completed / Skipped
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('Patient', backref=db.backref(
        'care_journey_events', lazy=True, order_by='CareJourneyEvent.stage_order'))
    assigned_doctor = db.relationship('Doctor', foreign_keys=[assigned_doctor_id])

class DischargeChecklist(db.Model):
    """One row per patient per required discharge task (6 fixed tasks, see
    DISCHARGE_TASKS in orchestrator_engine.py). Readiness % is computed live
    from these rows, never stored, so it can never go stale."""
    __tablename__ = 'discharge_checklist'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    task_name = db.Column(db.String(100), nullable=False)
    is_complete = db.Column(db.Boolean, default=False)
    completed_at = db.Column(db.DateTime, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('Patient', backref=db.backref('discharge_checklist', lazy=True))

class FollowUp(db.Model):
    """A real scheduled post-discharge follow-up. 'Missed' can be derived
    live (scheduled_date has passed, status still Scheduled) without needing
    a background job, so the stored status only needs updating when staff
    actually complete or cancel one."""
    __tablename__ = 'follow_ups'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=True)
    scheduled_date = db.Column(db.Date, nullable=False)
    purpose = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(20), default='Scheduled')  # Scheduled / Completed / Missed / Cancelled
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    patient = db.relationship('Patient', backref=db.backref('follow_ups', lazy=True))
    doctor = db.relationship('Doctor', foreign_keys=[doctor_id])

def seed_care_journey(patient):
    """Creates the 8 fixed CareJourneyEvent rows for a patient if they don't
    already exist. Admission and Doctor Assignment start pre-completed since
    those facts are already true the moment a patient record is created."""
    if CareJourneyEvent.query.filter_by(patient_id=patient.id).first():
        return
    now = datetime.utcnow()
    for order, name, dept in CARE_STAGES:
        pre_done = order <= 2
        db.session.add(CareJourneyEvent(
            patient_id=patient.id, stage_name=name, stage_order=order, department=dept,
            assigned_doctor_id=patient.doctor_id,
            status='Completed' if pre_done else ('In Progress' if order == 3 else 'Pending'),
            started_at=now if (pre_done or order == 3) else None,
            completed_at=now if pre_done else None
        ))
    db.session.commit()

def seed_discharge_checklist(patient):
    """Creates the 6 fixed DischargeChecklist rows for a patient if missing."""
    if DischargeChecklist.query.filter_by(patient_id=patient.id).first():
        return
    for task in DISCHARGE_TASKS:
        db.session.add(DischargeChecklist(patient_id=patient.id, task_name=task))
    db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

def log_activity(action, category='General', icon='bi-activity', color='primary'):
    try:
        db.session.add(ActivityLog(action=action, category=category, icon=icon, color=color))
        db.session.commit()
    except Exception:
        db.session.rollback()

def get_dashboard_stats():
    return {
        'total_patients': Patient.query.filter_by(status='Admitted').count(),
        'total_doctors': Doctor.query.filter_by(status='Active').count(),
        'total_beds': Bed.query.count(),
        'available_beds': Bed.query.filter_by(status='Available').count(),
        'occupied_beds': Bed.query.filter_by(status='Occupied').count(),
        'low_stock_meds': Medicine.query.filter(Medicine.quantity <= Medicine.low_stock_threshold).count(),
        'today_attendance': Attendance.query.filter_by(date=date.today(), status='Present').count(),
    }
@app.route('/')
def index():
    return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            flash('Please fill in all fields.', 'danger')
            return render_template('login.html')
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            login_user(admin, remember=True)
            log_activity(f'Admin {username} logged in', 'System', 'bi-box-arrow-in-right', 'success')
            flash(f'Welcome back, {admin.username}!', 'success')
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    log_activity(f'Admin {current_user.username} logged out', 'System', 'bi-box-arrow-right', 'secondary')
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    stats = get_dashboard_stats()
    recent_activities = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(8).all()
    bed_chart = {
        'available': stats['available_beds'],
        'occupied': stats['occupied_beds'],
        'maintenance': Bed.query.filter_by(status='Maintenance').count()
    }
    monthly_data, month_labels = [], []
    for i in range(5, -1, -1):
        month_num = ((date.today().month - i - 1) % 12) + 1
        monthly_data.append(Patient.query.filter(extract('month', Patient.admission_date) == month_num).count())
        month_labels.append(calendar.month_abbr[month_num])
    med_categories = db.session.query(Medicine.category, func.count(Medicine.id)).group_by(Medicine.category).all()
    return render_template('dashboard.html', stats=stats, recent_activities=recent_activities,
        bed_chart=bed_chart, monthly_data=monthly_data, month_labels=month_labels, med_categories=med_categories)

@app.route('/inventory')
@login_required
def inventory():
    search = request.args.get('search', '').strip()
    category_filter = request.args.get('category', '').strip()
    query = Medicine.query
    if search: query = query.filter(Medicine.name.ilike(f'%{search}%'))
    if category_filter: query = query.filter_by(category=category_filter)
    medicines = query.order_by(Medicine.name).all()
    categories = [c[0] for c in db.session.query(Medicine.category).distinct().all() if c[0]]
    return render_template('inventory.html', medicines=medicines, categories=categories,
        low_stock_count=sum(1 for m in medicines if m.is_low_stock),
        expired_count=sum(1 for m in medicines if m.is_expired),
        search=search, category_filter=category_filter)

@app.route('/inventory/add', methods=['POST'])
@login_required
def add_medicine():
    try:
        expiry_str = request.form.get('expiry_date')
        m = Medicine(
            name=request.form['name'].strip(),
            category=request.form.get('category', '').strip(),
            manufacturer=request.form.get('manufacturer', '').strip(),
            quantity=int(request.form.get('quantity', 0)),
            unit=request.form.get('unit', 'Tablets').strip(),
            price_per_unit=float(request.form.get('price_per_unit', 0.0)),
            low_stock_threshold=int(request.form.get('low_stock_threshold', 50)),
            expiry_date=datetime.strptime(expiry_str, '%Y-%m-%d').date() if expiry_str else None,
            description=request.form.get('description', '').strip(),
        )
        db.session.add(m)
        db.session.commit()
        log_activity(f'Medicine added: {m.name}', 'Medicine', 'bi-capsule', 'success')
        flash(f'Medicine "{m.name}" added successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('inventory'))

@app.route('/inventory/edit/<int:id>', methods=['POST'])
@login_required
def edit_medicine(id):
    m = Medicine.query.get_or_404(id)
    try:
        expiry_str = request.form.get('expiry_date')
        m.name = request.form['name'].strip()
        m.category = request.form.get('category', '').strip()
        m.manufacturer = request.form.get('manufacturer', '').strip()
        m.quantity = int(request.form.get('quantity', 0))
        m.unit = request.form.get('unit', 'Tablets').strip()
        m.price_per_unit = float(request.form.get('price_per_unit', 0.0))
        m.low_stock_threshold = int(request.form.get('low_stock_threshold', 50))
        m.expiry_date = datetime.strptime(expiry_str, '%Y-%m-%d').date() if expiry_str else None
        m.description = request.form.get('description', '').strip()
        m.updated_at = datetime.utcnow()
        db.session.commit()
        log_activity(f'Medicine updated: {m.name}', 'Medicine', 'bi-pencil', 'warning')
        flash(f'Medicine "{m.name}" updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('inventory'))

@app.route('/inventory/delete/<int:id>', methods=['POST'])
@login_required
def delete_medicine(id):
    m = Medicine.query.get_or_404(id)
    try:
        name = m.name
        db.session.delete(m)
        db.session.commit()
        log_activity(f'Medicine deleted: {name}', 'Medicine', 'bi-trash', 'danger')
        flash(f'Medicine "{name}" deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('inventory'))

@app.route('/beds')
@login_required
def beds():
    search = request.args.get('search', '').strip()
    ward_filter = request.args.get('ward', '').strip()
    query = Bed.query
    if search: query = query.filter(Bed.bed_number.ilike(f'%{search}%'))
    if ward_filter: query = query.filter_by(ward=ward_filter)
    all_beds = query.order_by(Bed.bed_number).all()
    wards = [w[0] for w in db.session.query(Bed.ward).distinct().all() if w[0]]
    unassigned_patients = Patient.query.filter_by(status='Admitted', bed_id=None).all()
    stats = {
        'available': Bed.query.filter_by(status='Available').count(),
        'occupied': Bed.query.filter_by(status='Occupied').count(),
        'maintenance': Bed.query.filter_by(status='Maintenance').count(),
        'total': Bed.query.count(),
    }
    return render_template('beds.html', beds=all_beds, wards=wards, stats=stats,
        unassigned_patients=unassigned_patients, search=search, ward_filter=ward_filter)

@app.route('/beds/add', methods=['POST'])
@login_required
def add_bed():
    try:
        bed = Bed(
            bed_number=request.form['bed_number'].strip(),
            ward=request.form['ward'].strip(),
            bed_type=request.form.get('bed_type', 'General').strip(),
            status=request.form.get('status', 'Available').strip(),
        )
        db.session.add(bed)
        db.session.commit()
        log_activity(f'Bed added: {bed.bed_number}', 'Bed', 'bi-hospital', 'primary')
        flash(f'Bed {bed.bed_number} added.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('beds'))

@app.route('/beds/assign', methods=['POST'])
@login_required
def assign_bed():
    bed = Bed.query.get_or_404(request.form.get('bed_id'))
    patient = Patient.query.get_or_404(request.form.get('patient_id'))
    try:
        bed.status = 'Occupied'
        patient.bed_id = bed.id
        db.session.commit()
        log_activity(f'Bed {bed.bed_number} assigned to {patient.name}', 'Bed', 'bi-person-fill', 'info')
        flash(f'Bed {bed.bed_number} assigned to {patient.name}.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('beds'))

@app.route('/beds/discharge/<int:bed_id>', methods=['POST'])
@login_required
def discharge_bed(bed_id):
    bed = Bed.query.get_or_404(bed_id)
    try:
        patient = Patient.query.filter_by(bed_id=bed_id, status='Admitted').first()
        if patient:
            patient.status = 'Discharged'
            patient.discharge_date = date.today()
            patient.bed_id = None
            log_activity(f'{patient.name} discharged from {bed.bed_number}', 'Patient', 'bi-person-dash', 'warning')
        bed.status = 'Available'
        db.session.commit()
        flash(f'Bed {bed.bed_number} is now available.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('beds'))

@app.route('/beds/update-status/<int:bed_id>', methods=['POST'])
@login_required
def update_bed_status(bed_id):
    bed = Bed.query.get_or_404(bed_id)
    try:
        bed.status = request.form.get('status', bed.status)
        db.session.commit()
        flash(f'Bed {bed.bed_number} status updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('beds'))
@app.route('/doctors')
@login_required
def doctors():
    search = request.args.get('search', '').strip()
    dept_filter = request.args.get('department', '').strip()
    query = Doctor.query
    if search: query = query.filter(Doctor.name.ilike(f'%{search}%'))
    if dept_filter: query = query.filter_by(department=dept_filter)
    all_doctors = query.order_by(Doctor.name).all()
    departments = [d[0] for d in db.session.query(Doctor.department).distinct().all() if d[0]]
    today_attendance = {a.doctor_id: a for a in Attendance.query.filter_by(date=date.today()).all()}
    return render_template('doctors.html', doctors=all_doctors, departments=departments,
        today_attendance=today_attendance, search=search, dept_filter=dept_filter, today=date.today())

@app.route('/doctors/add', methods=['POST'])
@login_required
def add_doctor():
    try:
        d = Doctor(
            name=request.form['name'].strip(),
            specialization=request.form['specialization'].strip(),
            department=request.form['department'].strip(),
            phone=request.form.get('phone', '').strip(),
            email=request.form.get('email', '').strip(),
            experience_years=int(request.form.get('experience_years', 0)),
        )
        db.session.add(d)
        db.session.commit()
        log_activity(f'Doctor added: Dr. {d.name}', 'Doctor', 'bi-person-badge', 'primary')
        flash(f'Dr. {d.name} added.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('doctors'))

@app.route('/doctors/edit/<int:id>', methods=['POST'])
@login_required
def edit_doctor(id):
    d = Doctor.query.get_or_404(id)
    try:
        d.name = request.form['name'].strip()
        d.specialization = request.form['specialization'].strip()
        d.department = request.form['department'].strip()
        d.phone = request.form.get('phone', '').strip()
        d.email = request.form.get('email', '').strip()
        d.experience_years = int(request.form.get('experience_years', 0))
        d.status = request.form.get('status', 'Active').strip()
        db.session.commit()
        log_activity(f'Doctor updated: Dr. {d.name}', 'Doctor', 'bi-pencil', 'warning')
        flash(f'Dr. {d.name} updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('doctors'))

@app.route('/doctors/attendance', methods=['POST'])
@login_required
def mark_attendance():
    doctor_id = request.form.get('doctor_id')
    status = request.form.get('status', 'Present')
    try:
        existing = Attendance.query.filter_by(doctor_id=doctor_id, date=date.today()).first()
        if existing:
            existing.status = status
            existing.check_in_time = request.form.get('check_in_time', '')
            existing.notes = request.form.get('notes', '')
        else:
            db.session.add(Attendance(
                doctor_id=doctor_id, status=status,
                check_in_time=request.form.get('check_in_time', ''),
                notes=request.form.get('notes', '')
            ))
        db.session.commit()
        d = Doctor.query.get(doctor_id)
        log_activity(f'Attendance {status}: Dr. {d.name}', 'Doctor', 'bi-calendar-check', 'info')
        flash(f'Attendance marked as {status}.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('doctors'))

@app.route('/patients')
@login_required
def patients():
    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', '').strip()
    query = Patient.query
    if search: query = query.filter(Patient.name.ilike(f'%{search}%'))
    if status_filter: query = query.filter_by(status=status_filter)
    all_patients = query.order_by(Patient.created_at.desc()).all()
    all_doctors = Doctor.query.filter_by(status='Active').all()
    stats = {
        'admitted': Patient.query.filter_by(status='Admitted').count(),
        'discharged': Patient.query.filter_by(status='Discharged').count(),
        'total': Patient.query.count(),
    }
    return render_template('patients.html', patients=all_patients, doctors=all_doctors,
        stats=stats, search=search, status_filter=status_filter)

@app.route('/patients/add', methods=['POST'])
@login_required
def add_patient():
    try:
        p = Patient(
            name=request.form['name'].strip(),
            age=int(request.form.get('age', 0)),
            gender=request.form.get('gender', '').strip(),
            phone=request.form.get('phone', '').strip(),
            address=request.form.get('address', '').strip(),
            disease=request.form.get('disease', '').strip(),
            medical_history=request.form.get('medical_history', '').strip(),
            doctor_id=request.form.get('doctor_id') or None,
            admission_date=date.today(),
        )
        db.session.add(p)
        db.session.commit()
        seed_care_journey(p)
        seed_discharge_checklist(p)
        log_activity(f'Patient registered: {p.name}', 'Patient', 'bi-person-plus', 'success')
        flash(f'Patient {p.name} registered.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('patients'))

@app.route('/patients/edit/<int:id>', methods=['POST'])
@login_required
def edit_patient(id):
    p = Patient.query.get_or_404(id)
    try:
        p.name = request.form['name'].strip()
        p.age = int(request.form.get('age', 0))
        p.gender = request.form.get('gender', '').strip()
        p.phone = request.form.get('phone', '').strip()
        p.address = request.form.get('address', '').strip()
        p.disease = request.form.get('disease', '').strip()
        p.medical_history = request.form.get('medical_history', '').strip()
        p.doctor_id = request.form.get('doctor_id') or None
        p.status = request.form.get('status', 'Admitted').strip()
        if p.status == 'Discharged' and not p.discharge_date:
            p.discharge_date = date.today()
        db.session.commit()
        log_activity(f'Patient updated: {p.name}', 'Patient', 'bi-pencil', 'warning')
        flash(f'Patient {p.name} updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('patients'))

@app.route('/patients/<int:id>/health-twin')
@login_required
def patient_health_twin(id):
    patient = Patient.query.get_or_404(id)
    return render_template('health_twin.html', patient=patient)

@app.route('/patients/<int:id>/health-twin/analyze', methods=['POST'])
@login_required
def analyze_health_twin(id):
    patient = Patient.query.get_or_404(id)

    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Add GEMINI_API_KEY to your .env file and restart the app.'}), 503

    try:
        previous = HealthTwinAnalysis.query.filter_by(patient_id=patient.id) \
            .order_by(HealthTwinAnalysis.created_at.desc()).first()

        previous_context = ''
        if previous:
            previous_context = (
                f"\nPREVIOUS ANALYSIS on file (from {previous.created_at.strftime('%d %b %Y')}): "
                f"overall score was {previous.overall_score}/100 ({previous.overall_status}). "
                f"Use this as a baseline so organ 'trend' values reflect real change "
                f"(Improving / Stable / Declining / Monitor) rather than being guessed independently."
            )

        prompt = f"""You are the Mediora AI Health Twin engine, used by hospital staff as a clinical decision-support illustration. This is not a diagnosis and must not replace physician judgement.

PATIENT RECORD:
Name: {patient.name}
Age: {patient.age if patient.age else 'unknown'}
Gender: {patient.gender or 'unknown'}
Current diagnosis: {patient.disease or 'None recorded'}
Medical history: {patient.medical_history or 'None recorded'}
Admission date: {patient.admission_date.strftime('%d %b %Y') if patient.admission_date else 'unknown'}
Status: {patient.status}
{previous_context}

Return ONLY a valid JSON object, no markdown code fences, no commentary before or after, with EXACTLY this structure:

{{
  "overall_score": <integer 0-100>,
  "overall_status": "<2-4 word status, e.g. 'Good Health' or 'Needs Attention'>",
  "summary": "<2-3 sentence plain-language summary of the patient's overall condition>",
  "risk_factors": ["<short risk factor>", "..."],
  "immediate_actions": ["<short recommended action>", "..."],
  "organs": [
    {{"icon":"\U0001FAC0","name":"Heart","score":<0-100>,"status":"<Good|Moderate|Needs Attention>","trend":"<Improving|Stable|Declining|Monitor>","risk":"<Low|Medium|High>","observation":"<1-2 sentence observation>","recommendation":"<1-2 sentence recommendation>"}},
    {{"icon":"\U0001FAC1","name":"Lungs","score":<0-100>,"status":"...","trend":"...","risk":"...","observation":"...","recommendation":"..."}},
    {{"icon":"\U0001F9E0","name":"Brain & Nervous System","score":<0-100>,"status":"...","trend":"...","risk":"...","observation":"...","recommendation":"..."}},
    {{"icon":"\U0001FA78","name":"Kidneys","score":<0-100>,"status":"...","trend":"...","risk":"...","observation":"...","recommendation":"..."}},
    {{"icon":"\U0001F37D\uFE0F","name":"Liver & Digestion","score":<0-100>,"status":"...","trend":"...","risk":"...","observation":"...","recommendation":"..."}},
    {{"icon":"\u26A1","name":"Metabolic Health","score":<0-100>,"status":"...","trend":"...","risk":"...","observation":"...","recommendation":"..."}}
  ],
  "whatif": {{
    "lose_weight": "<1 sentence projected effect>",
    "better_sleep": "<1 sentence projected effect>",
    "exercise": "<1 sentence projected effect>",
    "quit_smoking": "<1 sentence projected effect>"
  }}
}}

Base every score and observation on the diagnosis and medical history given. If information for a category is missing, infer a reasonable general-population baseline and note that briefly in that organ's observation. Keep language clinical but understandable to hospital staff."""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        raw_text = (response.text or '').strip()
        cleaned = re.sub(r'^```json\s*|^```\s*|```\s*$', '', raw_text, flags=re.MULTILINE).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not match:
                return jsonify({'error': 'AI returned an unreadable response. Please try again.'}), 502
            data = json.loads(match.group(0))

        required_keys = {'overall_score', 'overall_status', 'summary', 'risk_factors',
                          'immediate_actions', 'organs', 'whatif'}
        missing = required_keys - data.keys()
        if missing:
            return jsonify({'error': f'AI response was incomplete (missing: {", ".join(missing)}). Please try again.'}), 502
        if not isinstance(data['organs'], list) or len(data['organs']) == 0:
            return jsonify({'error': 'AI response contained no organ data. Please try again.'}), 502

        try:
            data['overall_score'] = max(0, min(100, int(data['overall_score'])))
        except (TypeError, ValueError):
            data['overall_score'] = 50

        record = HealthTwinAnalysis(
            patient_id=patient.id,
            overall_score=data['overall_score'],
            overall_status=str(data.get('overall_status', ''))[:50],
            analysis_json=json.dumps(data)
        )
        db.session.add(record)
        db.session.commit()

        log_activity(f'AI Health Twin generated for {patient.name}', 'AI Twin', 'bi-stars', 'info')

        return jsonify(data)

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/reports')
@login_required
def reports():
    bed_stats = {
        'available': Bed.query.filter_by(status='Available').count(),
        'occupied': Bed.query.filter_by(status='Occupied').count(),
        'maintenance': Bed.query.filter_by(status='Maintenance').count(),
    }
    med_by_category = db.session.query(Medicine.category, func.sum(Medicine.quantity)).group_by(Medicine.category).all()
    docs_by_dept = db.session.query(Doctor.department, func.count(Doctor.id)).group_by(Doctor.department).all()
    attendance_summary = db.session.query(Attendance.status, func.count(Attendance.id)).group_by(Attendance.status).all()
    patient_stats = {
        'admitted': Patient.query.filter_by(status='Admitted').count(),
        'discharged': Patient.query.filter_by(status='Discharged').count(),
        'male': Patient.query.filter_by(gender='Male').count(),
        'female': Patient.query.filter_by(gender='Female').count(),
    }
    top_medicines = Medicine.query.order_by(Medicine.quantity.desc()).limit(5).all()
    low_stock_meds = Medicine.query.filter(Medicine.quantity <= Medicine.low_stock_threshold).all()
    return render_template('reports.html', bed_stats=bed_stats, med_by_category=med_by_category,
        docs_by_dept=docs_by_dept, attendance_summary=attendance_summary, patient_stats=patient_stats,
        top_medicines=top_medicines, low_stock_meds=low_stock_meds)

@app.route('/chatbot')
@login_required
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot/ask', methods=['POST'])
@login_required
def chatbot_ask():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    if not GEMINI_API_KEY:
        return jsonify({'response': 'Gemini API key not configured.'})
    try:
        stats = get_dashboard_stats()
        low_stock = Medicine.query.filter(Medicine.quantity <= Medicine.low_stock_threshold).all()
        context = f"""You are Mediora Health Assistant for a government hospital in India.
Hospital: Patients={stats['total_patients']}, Doctors={stats['total_doctors']},
Beds={stats['available_beds']}/{stats['total_beds']} available,
Low Stock={', '.join([m.name for m in low_stock]) if low_stock else 'None'}.
Answer in Hindi if user writes Hindi. Be concise. Recommend doctor for medical advice."""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"{context}\n\nUser: {user_message}")
        return jsonify({'response': response.text or 'Could not generate response.'})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/ai-insights')
@login_required
def ai_insights():
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API not configured'}), 503
    try:
        stats = get_dashboard_stats()
        low_stock = Medicine.query.filter(Medicine.quantity <= Medicine.low_stock_threshold).all()
        expired = [m for m in Medicine.query.all() if m.is_expired]
        occ = round((stats['occupied_beds'] / stats['total_beds'] * 100) if stats['total_beds'] > 0 else 0, 1)
        prompt = f"""Hospital: Patients={stats['total_patients']}, Beds={stats['available_beds']}/{stats['total_beds']},
Occupancy={occ}%, LowStock={', '.join([m.name for m in low_stock]) if low_stock else 'None'},
Expired={len(expired)}, DoctorsPresent={stats['today_attendance']}/{stats['total_doctors']}.
Give 5 insights as JSON: [{{"title":"...","description":"...","priority":"High/Medium/Low","category":"..."}}]
Return ONLY valid JSON."""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        text = re.sub(r'```json|```', '', response.text.strip()).strip()
        return jsonify({'insights': json.loads(text)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sentinel')
@login_required
def sentinel():
    return render_template('sentinel.html')

@app.route('/sentinel/scan', methods=['POST'])
@login_required
def sentinel_scan():
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Add GEMINI_API_KEY to your .env file and restart the app.'}), 503

    def call_gemini(prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text or ''

    try:
        result = sentinel_engine.run_sentinel_scan(
            Patient, Doctor, Bed, Medicine, Attendance, call_gemini
        )
        log_activity('Sentinel scan executed', 'Sentinel', 'bi-shield-check', 'info')
        return jsonify(result)
    except ValueError as e:
        # raised by sentinel_engine.parse_sentinel_ai_response on bad/unparseable AI output
        return jsonify({'error': str(e)}), 502
    except Exception as e:
        return jsonify({'error': f'Sentinel scan failed: {str(e)}'}), 500

@app.route('/orchestrator')
@login_required
def orchestrator():
    return render_template('orchestrator.html')

def _build_patient_payload(patient):
    """Gathers one admitted patient's real rows into the flat dict shape
    orchestrator_engine.run_orchestrator_scan() expects. Lazily seeds rows
    for patients created before this feature existed, so nothing breaks."""
    seed_care_journey(patient)
    seed_discharge_checklist(patient)

    events = CareJourneyEvent.query.filter_by(patient_id=patient.id).order_by(CareJourneyEvent.stage_order).all()
    event_dicts = [{
        'stage_name': e.stage_name, 'stage_order': e.stage_order, 'department': e.department,
        'doctor_name': e.assigned_doctor.name if e.assigned_doctor else None,
        'status': e.status, 'started_at': e.started_at, 'completed_at': e.completed_at,
    } for e in events]

    checklist = DischargeChecklist.query.filter_by(patient_id=patient.id).all()
    checklist_dicts = [{'id': c.id, 'task_name': c.task_name, 'is_complete': c.is_complete} for c in checklist]

    followups = FollowUp.query.filter_by(patient_id=patient.id).all()
    followup_dicts = [{
        'id': f.id, 'scheduled_date': f.scheduled_date, 'status': f.status,
        'purpose': f.purpose,
    } for f in followups]

    return {
        'patient_id': patient.id,
        'patient_name': patient.name,
        'journey_events': event_dicts,
        'checklist_items': checklist_dicts,
        'followups': followup_dicts,
    }

@app.route('/orchestrator/scan', methods=['POST'])
@login_required
def orchestrator_scan():
    if not GEMINI_API_KEY:
        return jsonify({'error': 'Gemini API key not configured. Add GEMINI_API_KEY to your .env file and restart the app.'}), 503

    def call_gemini(prompt):
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text or ''

    try:
        admitted = Patient.query.filter_by(status='Admitted').all()
        patients_raw = [_build_patient_payload(p) for p in admitted]
        result = orchestrator_engine.run_orchestrator_scan(patients_raw, call_gemini)
        log_activity('Orchestrator workflow scan executed', 'Orchestrator', 'bi-diagram-3', 'info')
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Orchestrator scan failed: {str(e)}'}), 500

@app.route('/orchestrator/patient/<int:id>/advance', methods=['POST'])
@login_required
def orchestrator_advance_stage(id):
    patient = Patient.query.get_or_404(id)
    try:
        seed_care_journey(patient)
        events = CareJourneyEvent.query.filter_by(patient_id=patient.id).order_by(CareJourneyEvent.stage_order).all()
        in_progress = next((e for e in events if e.status == 'In Progress'), None)
        if not in_progress:
            return jsonify({'error': 'No stage is currently in progress for this patient.'}), 400

        in_progress.status = 'Completed'
        in_progress.completed_at = datetime.utcnow()

        next_stage = next((e for e in events if e.stage_order == in_progress.stage_order + 1), None)
        if next_stage:
            next_stage.status = 'In Progress'
            next_stage.started_at = datetime.utcnow()

        db.session.commit()
        log_activity(f'{patient.name} advanced past {in_progress.stage_name}', 'Orchestrator', 'bi-arrow-right-circle', 'success')
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/orchestrator/checklist/<int:item_id>/toggle', methods=['POST'])
@login_required
def orchestrator_toggle_checklist(item_id):
    item = DischargeChecklist.query.get_or_404(item_id)
    try:
        item.is_complete = not item.is_complete
        item.completed_at = datetime.utcnow() if item.is_complete else None
        db.session.commit()
        return jsonify({'success': True, 'is_complete': item.is_complete})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/orchestrator/patient/<int:id>/followup', methods=['POST'])
@login_required
def orchestrator_add_followup(id):
    patient = Patient.query.get_or_404(id)
    try:
        scheduled_date_str = request.form.get('scheduled_date') or request.json.get('scheduled_date')
        purpose = request.form.get('purpose') or (request.json.get('purpose') if request.is_json else None)
        scheduled_date = datetime.strptime(scheduled_date_str, '%Y-%m-%d').date()

        fu = FollowUp(patient_id=patient.id, doctor_id=patient.doctor_id,
                       scheduled_date=scheduled_date, purpose=purpose, status='Scheduled')
        db.session.add(fu)
        db.session.commit()
        log_activity(f'Follow-up scheduled for {patient.name}', 'Orchestrator', 'bi-calendar-check', 'info')
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def seed_database():
    if not Admin.query.first():
        a = Admin(username='admin', email='admin@mediora.health', hospital_name='Mediora PHC - Greater Noida')
        a.set_password('admin123')
        db.session.add(a)

    if not Doctor.query.first():
        db.session.add_all([
            Doctor(name='Priya Sharma', specialization='General Physician', department='General Medicine', phone='9876543210', experience_years=8),
            Doctor(name='Rajesh Kumar', specialization='Cardiologist', department='Cardiology', phone='9876543211', experience_years=12),
            Doctor(name='Sunita Patel', specialization='Pediatrician', department='Pediatrics', phone='9876543212', experience_years=6),
            Doctor(name='Amit Singh', specialization='Orthopedic', department='Orthopedics', phone='9876543213', experience_years=10),
            Doctor(name='Meera Verma', specialization='Gynecologist', department='Gynecology', phone='9876543214', experience_years=9),
        ])

    if not Bed.query.first():
        beds_data, counter = [], 1
        for ward, btype in [('General Ward A','General'),('General Ward B','General'),('ICU','ICU'),('Pediatric Ward','General'),('Private Room','Private')]:
            for _ in range(5):
                beds_data.append(Bed(bed_number=f'B{counter:03d}', ward=ward, bed_type=btype,
                    status='Available' if counter % 3 != 0 else 'Occupied'))
                counter += 1
        db.session.add_all(beds_data)

    if not Medicine.query.first():
        db.session.add_all([
            Medicine(name='Paracetamol 500mg', category='Analgesic', manufacturer='Sun Pharma', quantity=500, unit='Tablets', price_per_unit=2.5, low_stock_threshold=100, expiry_date=date.today()+timedelta(days=365)),
            Medicine(name='Amoxicillin 250mg', category='Antibiotic', manufacturer='Cipla', quantity=200, unit='Capsules', price_per_unit=8.0, low_stock_threshold=50, expiry_date=date.today()+timedelta(days=180)),
            Medicine(name='Metformin 500mg', category='Antidiabetic', manufacturer="Dr. Reddy's", quantity=30, unit='Tablets', price_per_unit=5.0, low_stock_threshold=50, expiry_date=date.today()+timedelta(days=270)),
            Medicine(name='Amlodipine 5mg', category='Antihypertensive', manufacturer='Lupin', quantity=150, unit='Tablets', price_per_unit=6.5, low_stock_threshold=50, expiry_date=date.today()+timedelta(days=300)),
            Medicine(name='ORS Sachets', category='Electrolyte', manufacturer='Govt Supply', quantity=20, unit='Sachets', price_per_unit=3.0, low_stock_threshold=30, expiry_date=date.today()+timedelta(days=400)),
            Medicine(name='Dolo 650', category='Analgesic', manufacturer='Micro Labs', quantity=800, unit='Tablets', price_per_unit=2.0, low_stock_threshold=100, expiry_date=date.today()+timedelta(days=500)),
            Medicine(name='Pantoprazole 40mg', category='Antacid', manufacturer='Zydus', quantity=45, unit='Tablets', price_per_unit=7.0, low_stock_threshold=50, expiry_date=date.today()+timedelta(days=220)),
            Medicine(name="Ringer's Lactate", category='IV Fluid', manufacturer='B. Braun', quantity=15, unit='Bags', price_per_unit=65.0, low_stock_threshold=20, expiry_date=date.today()+timedelta(days=600)),
        ])

    db.session.commit()
    print("✅ Database ready!")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_database()
    app.run(debug=True, host='0.0.0.0', port=5000)