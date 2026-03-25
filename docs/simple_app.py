from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def simple_form():
    name = None
    email = None
    message = None
    
    if request.method == 'POST':
        # Extract data from the form
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        # Here you could save to a database or perform logic
        print(f"Received Submission: {name} ({email}) - {message}")

    return render_template('simple_form.html', name=name, email=email, message=message)

if __name__ == '__main__':
    print("Starting simple Flask app on http://127.0.0.1:5001")
    # Using port 5001 to avoid conflict with potential existing app on 5000
    app.run(debug=True, port=5001)
