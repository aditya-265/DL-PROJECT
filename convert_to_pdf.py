# simple_pdf_converter.py
import subprocess
import os

def create_simple_pdf():
    """Create PDF using system tools"""
    
    content = """
# Mango Leaf Disease Classification Report

**Group Members:**
- Aditya Gupta (102215265)
- Madhav Gupta (102395010)
- Sakshi Rana (102215293)
- Diksha Sood (102395018)
[Rest of your report content...]
"""
    
    # Save as text file
    with open('report.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Report saved as report.txt")
    print("💡 You can now copy this to Word and save as PDF")

if __name__ == "__main__":
    create_simple_pdf()