from flask import Flask, render_template_string
import os
app = Flask(__name__)

# HTML template with navigation links and styling
page_template = """
<!doctype html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 40px;
        }
        nav {
            position: absolute;
            top: 10px;
            right: 10px;
        }
        nav a {
            padding: 10px;
            text-decoration: none;
            color: #333;
            border: 1px solid #ccc;
            margin-left: 5px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        nav a:hover {
            background-color: #f5f5f5;
        }
        form {
            margin-top: 50px;
        }
        input[type="submit"] {
            padding: 10px 20px;
            background-color: #007BFF;
            color: #fff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        input[type="submit"]:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <nav>
        <a href="{{ url_for('home') }}">Home</a>
        <a href="{{ url_for('news') }}">News</a>
        <a href="{{ url_for('topic') }}">Topic</a>
        <a href="{{ url_for('companion') }}">Companion</a>
        <a href="{{ url_for('about') }}">About</a>
        <a href="{{ url_for('gesture') }}">Gesture</a>
    </nav>
    <form action="{{ url_for('print_message', page_name=title.lower()) }}" method="post">
        <input type="submit" value="start">
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(page_template, title="Home")

@app.route('/news')
def news():
    return render_template_string(page_template, title="News")

@app.route('/topic')
def topic():
    return render_template_string(page_template, title="Topic")

@app.route('/companion')
def companion():
    return render_template_string(page_template, title="Companion")

@app.route('/about')
def about():
    return render_template_string(page_template, title="About")

@app.route('/gesture')
def gesture():
    return render_template_string(page_template, title="Gesture")

@app.route('/print/<page_name>', methods=['POST'])
def print_message(page_name):
    messages = {
        "home": "This is the Home page!",
        "news": "python news.py",
        "topic": "python topic.py",
        "companion": "python companion.py",
        "about": "Learn more about us!",
        "gesture": "streamlit run app.py"
    }
    os.system(messages.get(page_name, "Unknown page"))
    page_name=page_name
    return f"Message printed from {page_name.capitalize()} page!", 200

if __name__ == '__main__':
    app.run(debug=True)
