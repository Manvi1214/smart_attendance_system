import os
import csv
import io
import datetime
from flask import Blueprint, send_file, request, jsonify, session, Response
from database.models import AttendanceModel, StudentModel, SessionModel
from database.db_init import db
from config import Config
# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
from reportlab.lib.pagesizes import letter
# pyrefly: ignore [missing-import]
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
# pyrefly: ignore [missing-import]
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# pyrefly: ignore [missing-import]
from reportlab.lib import colors
# pyrefly: ignore [missing-import]
from bson import ObjectId

report_bp = Blueprint('report', __name__)

@report_bp.route('/reports/export/csv/<session_id>')
def export_csv(session_id):
    try:
        session_obj = db.sessions.find_one({"_id": ObjectId(session_id)})
        if not session_obj:
            return jsonify({"success": False, "message": "Session not found."}), 404
            
        attendance_logs = AttendanceModel.get_by_session(session_id)
        
        # Prepare CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header row
        writer.writerow(['Roll No', 'Name', 'Branch', 'Section', 'Date', 'Time', 'Status'])
        
        for record in attendance_logs:
            writer.writerow([
                record['student_roll'],
                record['student_name'],
                record['student_branch'],
                record['student_section'],
                record['date'],
                record['time'],
                record['status']
            ])
            
        filename = f"Attendance_{session_obj['subject']}_{session_obj['date']}.csv"
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@report_bp.route('/reports/export/excel/<session_id>')
def export_excel(session_id):
    try:
        session_obj = db.sessions.find_one({"_id": ObjectId(session_id)})
        if not session_obj:
            return jsonify({"success": False, "message": "Session not found."}), 404
            
        attendance_logs = AttendanceModel.get_by_session(session_id)
        
        # Build pandas DataFrame
        data = []
        for r in attendance_logs:
            data.append({
                "Roll No": r['student_roll'],
                "Student Name": r['student_name'],
                "Branch": r['student_branch'],
                "Section": r['student_section'],
                "Date": r['date'],
                "Time": r['time'],
                "Status": r['status']
            })
            
        df = pd.DataFrame(data)
        
        # Output excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            
        output.seek(0)
        filename = f"Attendance_{session_obj['subject']}_{session_obj['date']}.xlsx"
        
        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@report_bp.route('/reports/export/pdf/<student_id>')
def export_pdf(student_id):
    try:
        student = db.students.find_one({"_id": ObjectId(student_id)})
        if not student:
            return jsonify({"success": False, "message": "Student not found."}), 404
            
        attendance_records = AttendanceModel.get_by_student(student["_id"])
        total_sessions = db.sessions.count_documents({})
        present_count = len(attendance_records)
        absent_count = max(0, total_sessions - present_count)
        percentage = (present_count / total_sessions * 100) if total_sessions > 0 else 0.0
        
        # Create PDF in-memory buffer
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter,
                                rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        
        styles = getSampleStyleSheet()
        
        # Custom styles for a clean design
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=24,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=15
        )
        
        text_style = ParagraphStyle(
            name='TextStyle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            textColor=colors.HexColor('#334155'),
            spaceAfter=6
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Smart Attendance System", title_style))
        story.append(Paragraph(f"<b>Student Attendance Report</b>", ParagraphStyle(
            name='SubTitle', parent=styles['Heading2'], fontSize=16, spaceAfter=20, textColor=colors.HexColor('#475569')
        )))
        story.append(Spacer(1, 10))
        
        # Profile Data Table
        profile_data = [
            [Paragraph(f"<b>Name:</b> {student['name']}", text_style), Paragraph(f"<b>Roll No:</b> {student['roll_no']}", text_style)],
            [Paragraph(f"<b>Branch:</b> {student['branch']}", text_style), Paragraph(f"<b>Section:</b> {student['section']}", text_style)],
            [Paragraph(f"<b>Total Sessions:</b> {total_sessions}", text_style), Paragraph(f"<b>Present:</b> {present_count} | <b>Absent:</b> {absent_count}", text_style)],
            [Paragraph(f"<b>Attendance Percentage:</b> {round(percentage, 1)}%", text_style), Paragraph(f"<b>Report Date:</b> {datetime.date.today().strftime('%B %d, %Y')}", text_style)]
        ]
        
        profile_table = Table(profile_data, colWidths=[260, 260])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
        ]))
        story.append(profile_table)
        story.append(Spacer(1, 20))
        
        # Detailed Records Table
        table_data = [['#', 'Subject', 'Date', 'Time', 'Status']]
        for i, record in enumerate(attendance_records, 1):
            table_data.append([
                str(i),
                record['subject'],
                record['date'],
                record['time'],
                record['status']
            ])
            
        record_table = Table(table_data, colWidths=[40, 200, 100, 100, 80])
        record_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
            ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,-1), 10),
            ('BOTTOMPADDING', (0,1), (-1,-1), 6),
        ]))
        
        story.append(Paragraph("<b>Attendance Record History</b>", styles['Heading3']))
        story.append(Spacer(1, 10))
        story.append(record_table)
        
        # Build Document
        doc.build(story)
        pdf_buffer.seek(0)
        
        filename = f"Attendance_Report_{student['roll_no']}_{student['name']}.pdf"
        return send_file(
            pdf_buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
