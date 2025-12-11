# GitHub Repository Setup Guide

This guide will help you create a GitHub repository and push your mall movement tracking project to GitHub.

## Prerequisites

1. **Git installed** on your system
   - Check: `git --version`
   - Download: https://git-scm.com/downloads

2. **GitHub account**
   - Sign up at: https://github.com

3. **GitHub CLI (optional but recommended)**
   - Download: https://cli.github.com/

---

## Step 1: Initialize Git Repository

Open your terminal/PowerShell in the project directory and run:

```bash
# Navigate to project directory (if not already there)
cd "C:\Users\hp\Desktop\mall-movement-tracking ml"

# Initialize git repository
git init

# Check status
git status
```

---

## Step 2: Configure Git (First Time Only)

If this is your first time using Git, configure your name and email:

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

## Step 3: Create .gitignore (Already Created)

The `.gitignore` file is already created in your project. It excludes:
- Python cache files
- Virtual environments
- Large data files
- Model files (.pkl)
- IDE files
- Environment variables

**Important:** Large files (data, models) are excluded. Only commit:
- Code files
- Configuration templates
- Documentation
- Small sample data (if needed)

---

## Step 4: Create GitHub Repository

### Option A: Using GitHub Website (Recommended for Beginners)

1. **Go to GitHub**: https://github.com
2. **Click the "+" icon** in the top right corner
3. **Select "New repository"**
4. **Fill in details:**
   - Repository name: `mall-movement-tracking`
   - Description: `ML project for tracking and analyzing customer movement patterns in malls`
   - Visibility: Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. **Click "Create repository"**

### Option B: Using GitHub CLI

```bash
# Login to GitHub
gh auth login

# Create repository
gh repo create mall-movement-tracking --public --description "ML project for tracking and analyzing customer movement patterns in malls"
```

---

## Step 5: Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status

# See a summary
git status --short
```

---

## Step 6: Make Initial Commit

```bash
# Create initial commit
git commit -m "Initial commit: Mall movement tracking ML project

- Feature engineering pipeline
- Classification, clustering, and forecasting models
- Streamlit dashboard
- FastAPI endpoints
- Comprehensive EDA and feature analysis notebooks
- Training scripts and utilities"
```

---

## Step 7: Connect to GitHub Repository

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/mall-movement-tracking.git

# Verify remote
git remote -v
```

**Alternative (SSH):**
```bash
git remote add origin git@github.com:YOUR_USERNAME/mall-movement-tracking.git
```

---

## Step 8: Push to GitHub

```bash
# Push to GitHub (first time)
git branch -M main
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a **Personal Access Token** (not your GitHub password)
  - Create token: https://github.com/settings/tokens
  - Select scopes: `repo` (full control)

---

## Step 9: Verify on GitHub

1. Go to: `https://github.com/YOUR_USERNAME/mall-movement-tracking`
2. Verify all files are uploaded
3. Check that large files (data, models) are NOT uploaded (as per .gitignore)

---

## Step 10: Update README.md (Optional but Recommended)

Make sure your `README.md` includes:
- Project description
- Installation instructions
- Usage guide
- Project structure

---

## Daily Workflow: Making Changes and Pushing

After making changes to your project:

```bash
# 1. Check what changed
git status

# 2. Add specific files or all changes
git add .
# OR add specific files:
# git add notebooks/01_EDA.ipynb

# 3. Commit with descriptive message
git commit -m "Description of changes"

# 4. Push to GitHub
git push
```

---

## Common Git Commands

```bash
# Check status
git status

# View changes
git diff

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature/new-feature

# Switch branches
git checkout main

# Merge branch
git merge feature/new-feature

# Pull latest changes
git pull

# View remote repositories
git remote -v

# Update remote URL (if needed)
git remote set-url origin https://github.com/YOUR_USERNAME/mall-movement-tracking.git
```

---

## Troubleshooting

### Issue: "Permission denied" when pushing
**Solution**: Use Personal Access Token instead of password

### Issue: "Large files" error
**Solution**: Check `.gitignore` - large files should be excluded

### Issue: "Remote origin already exists"
**Solution**: 
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/mall-movement-tracking.git
```

### Issue: "Failed to push some refs"
**Solution**: Pull first, then push
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

---

## Best Practices

1. **Commit often** with descriptive messages
2. **Don't commit**:
   - Large data files
   - Model files (.pkl)
   - Environment variables (.env)
   - API keys or secrets
3. **Use branches** for new features
4. **Write clear commit messages**
5. **Keep README.md updated**

---

## Quick Reference: Complete Setup (Copy-Paste)

```bash
# 1. Initialize
git init

# 2. Configure (first time only)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 3. Add files
git add .

# 4. Commit
git commit -m "Initial commit: Mall movement tracking ML project"

# 5. Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/mall-movement-tracking.git

# 6. Push
git branch -M main
git push -u origin main
```

---

## Next Steps

1. ✅ Repository created and connected
2. ✅ Initial code pushed
3. 📝 Update README.md with project details
4. 🔒 Add collaborators (if needed)
5. 📊 Set up GitHub Actions for CI/CD (optional)
6. 📦 Add releases/tags for versions (optional)

---

## Need Help?

- Git Documentation: https://git-scm.com/doc
- GitHub Guides: https://guides.github.com
- GitHub Support: https://support.github.com

