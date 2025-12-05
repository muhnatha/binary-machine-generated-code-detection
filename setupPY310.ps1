# untuk menginstall 3.10 dan membuat virtual environment baru dengan Python 3.10

# Install Python 3.10
winget install Python.Python.3.10 --silent

# Wait for environment update
Start-Sleep -Seconds 5

# Remove old venv
Remove-Item -Recurse -Force .\venv

# Create new venv with Python 3.10
py -3.10 -m venv venv

# Activate venv
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

#.\setup.ps1 
# untuk run script ini di PowerShell