from flask import Blueprint, request, render_template, redirect, url_for, session, flash, jsonify
from database.models import UserModel, StudentModel
from database.db_init import db

auth_bp = Blueprint('auth', __name__, template_folder='../templates')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # API request vs Form submit
        is_json = request.is_json
        data = request.get_json() if is_json else request.form
        
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            error_msg = "Please provide both email and password."
            if is_json:
                return jsonify({"success": False, "message": error_msg}), 400
            flash(error_msg, 'danger')
            return render_template('auth/login.html')
            
        user = UserModel.verify_user(email, password)
        if user:
            # Set session variables
            session['user_id'] = str(user['_id'])
            session['email'] = user['email']
            session['role'] = user['role']
            
            # If student, store student_id and name
            if user['role'] == 'student':
                student = StudentModel.get_by_user_id(user['_id'])
                if student:
                    session['student_id'] = str(student['_id'])
                    session['student_name'] = student['name']
                    
            success_msg = "Logged in successfully!"
            if is_json:
                return jsonify({
                    "success": True, 
                    "message": success_msg, 
                    "role": user['role'],
                    "redirect": url_for('teacher.dashboard') if user['role'] == 'teacher' else url_for('student.dashboard')
                })
                
            flash(success_msg, 'success')
            if user['role'] == 'teacher':
                return redirect(url_for('teacher.dashboard'))
            else:
                return redirect(url_for('student.dashboard'))
        else:
            error_msg = "Invalid email or password."
            if is_json:
                return jsonify({"success": False, "message": error_msg}), 401
            flash(error_msg, 'danger')
            return render_template('auth/login.html')
            
    return render_template('auth/login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'student')
        
        if not email or not password:
            flash("Please fill in all fields.", 'danger')
            return render_template('auth/register.html')
            
        # Create user
        result = UserModel.create_user(email, password, role)
        if "error" in result:
            flash(result["error"], 'danger')
            return render_template('auth/register.html')
            
        flash("Registration successful! You can now log in.", 'success')
        return redirect(url_for('auth.login'))
        
    return render_template('auth/register.html')

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", 'info')
    return redirect(url_for('auth.login'))
