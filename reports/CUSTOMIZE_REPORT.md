# 📝 How to Customize the PDF Report

This guide shows you how to add group member names, project details, and other information to the PDF report.

---

## 🎯 Quick Customization

### Step 1: Open the Generator Script
Open `reports/generate_report.py` in your editor.

### Step 2: Find the `get_project_info()` Function
Look for this function around **line 40-60**:

```python
def get_project_info():
    """Get project information - CUSTOMIZE THIS SECTION"""
    return {
        'project_name': 'Mall Movement Tracking',
        'version': '1.0.0',
        ...
    }
```

### Step 3: Update Your Information

Edit the following sections:

---

## 👥 Add Group Members

Find the `'group_members'` section and update with your team:

```python
'group_members': [
    {'name': 'John Doe', 'role': 'Lead Developer', 'email': 'john@example.com'},
    {'name': 'Jane Smith', 'role': 'Data Scientist', 'email': 'jane@example.com'},
    {'name': 'Bob Johnson', 'role': 'ML Engineer', 'email': 'bob@example.com'},
    # Add more members as needed
],
```

**Fields:**
- `name`: Full name of team member
- `role`: Their role or contribution (e.g., "Lead Developer", "Data Analyst")
- `email`: Contact email (optional)

---

## 🏫 Add Project Details

Update these fields:

```python
'institution': 'Your University/Institution Name',
'course': 'Machine Learning Course / CS 101',
'supervisor': 'Dr. Professor Name (if applicable)',
'project_duration': 'January 2024 - December 2024',
```

**Fields:**
- `institution`: Your school/university/company name
- `course`: Course name or code
- `supervisor`: Supervisor/advisor name (leave empty if not applicable)
- `project_duration`: Start and end dates

---

## 🛠️ Add Technologies

Update the technologies list:

```python
'technologies': [
    'Python', 
    'Scikit-learn', 
    'XGBoost', 
    'Streamlit', 
    'Pandas', 
    'NumPy',
    'Matplotlib',
    'Seaborn'
    # Add more as needed
]
```

---

## 📋 Complete Example

Here's a complete example with all fields filled:

```python
def get_project_info():
    """Get project information - CUSTOMIZE THIS SECTION"""
    return {
        'project_name': 'Mall Movement Tracking',
        'version': '1.0.0',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'description': 'ML-powered analytics for customer movement patterns in shopping malls',
        
        # Group Members
        'group_members': [
            {'name': 'Aman Kumar', 'role': 'Project Lead & ML Engineer', 'email': 'aman@example.com'},
            {'name': 'Sarah Johnson', 'role': 'Data Scientist', 'email': 'sarah@example.com'},
            {'name': 'Mike Chen', 'role': 'Frontend Developer', 'email': 'mike@example.com'},
        ],
        
        # Project Details
        'institution': 'University of Technology',
        'course': 'CS 401 - Machine Learning',
        'supervisor': 'Dr. Emily Watson',
        'project_duration': 'September 2024 - December 2024',
        
        # Technologies
        'technologies': [
            'Python 3.x',
            'Scikit-learn',
            'XGBoost',
            'Streamlit',
            'Pandas',
            'NumPy',
            'Matplotlib',
            'Seaborn',
            'Plotly'
        ]
    }
```

---

## 🔄 Regenerate PDF After Changes

After updating the information:

```bash
python reports/generate_report.py
```

This will create a new PDF with your updated information.

---

## 📄 What Gets Added to PDF

### Title Page
- Project name
- Institution (if provided)
- Course (if provided)
- Supervisor (if provided)
- Project duration (if provided)
- Version and date
- **Group members table** with names, roles, and emails
- Technologies used

### Footer
- Generation date
- **Team member names** (prepared by)

---

## 💡 Tips

1. **Leave fields empty** if not applicable:
   ```python
   'supervisor': '',  # Empty if no supervisor
   ```

2. **Add as many members** as needed:
   ```python
   'group_members': [
       {'name': 'Member 1', 'role': 'Role 1', 'email': 'email1@example.com'},
       {'name': 'Member 2', 'role': 'Role 2', 'email': 'email2@example.com'},
       # ... add more
   ],
   ```

3. **Customize descriptions**:
   ```python
   'description': 'Your custom project description here',
   ```

4. **Update version** if needed:
   ```python
   'version': '1.1.0',  # Update version number
   ```

---

## ✅ Checklist

Before generating PDF:
- [ ] Updated all group member names
- [ ] Added roles/contributions
- [ ] Added email addresses (optional)
- [ ] Updated institution name
- [ ] Added course information
- [ ] Added supervisor (if applicable)
- [ ] Updated project duration
- [ ] Updated technologies list
- [ ] Customized description (if needed)

---

## 🚀 Quick Command

After making changes, regenerate:

```bash
python reports/generate_report.py
```

The new PDF will be saved as: `reports/Project_Report_YYYYMMDD_HHMMSS.pdf`

---

## 📝 Example Output

After customization, your PDF will include:

**Title Page:**
- Project Title
- Institution: University of Technology
- Course: CS 401 - Machine Learning
- Supervisor: Dr. Emily Watson
- Project Duration: September 2024 - December 2024
- **Team Members Table:**
  | Name | Role/Contribution | Email |
  |------|-------------------|-------|
  | Aman Kumar | Project Lead & ML Engineer | aman@example.com |
  | Sarah Johnson | Data Scientist | sarah@example.com |
  | Mike Chen | Frontend Developer | mike@example.com |
- Technologies: Python, Scikit-learn, XGBoost, etc.

**Footer:**
- Prepared by: Aman Kumar, Sarah Johnson, Mike Chen

---

**Need Help?** Check the `generate_report.py` file for comments and examples!

