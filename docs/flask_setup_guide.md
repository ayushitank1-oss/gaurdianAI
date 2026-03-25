# Getting Started with Flask

Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy, with the ability to scale up to complex applications.

## 1. Project Setup

### Create a Project Directory
```powershell
mkdir Flask_Tutorial
cd Flask_Tutorial
```

### Setup a Virtual Environment
It's recommended to use a virtual environment to manage dependencies for your project.
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Install Flask
```powershell
pip install Flask
```

## 2. Directory Structure
A typical basic Flask project looks like this:
```
project/
├── app.py           # Main application logic
├── static/          # CSS, JS, and Images
│   └── style.css
└── templates/       # HTML files (Jinja2)
    └── index.html
```

## 3. Creating Your First App
In `app.py`, you initialize the Flask app and define routes:
```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)
```

## 4. Handling Forms
To handle forms, you need to import `request` and check for the `POST` method:
```python
from flask import request

@app.route('/form', methods=['GET', 'POST'])
def my_form():
    if request.method == 'POST':
        # Process data
        name = request.form.get('user_name')
        return f"Hello, {name}!"
    return render_template('my_form.html')
```
