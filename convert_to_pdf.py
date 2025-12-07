# simple_pdf_converter.py
import subprocess
import os

def create_simple_pdf():
    """Create PDF using system tools"""
    
    content = """
# Mango Leaf Disease Classification Report

**Group Members:**
- Devit Shah (102217044)
- Samarth Kanwal (102217056) 
- Nayjot Singh (102217046)
- Pragun Sharma (102217043)

[Rest of your report content...]
"""
    
    # Save as text file
    with open('report.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Report saved as report.txt")
    print("💡 You can now copy this to Word and save as PDF")

if __name__ == "__main__":
    create_simple_pdf()