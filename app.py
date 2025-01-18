from flask import Flask
from src.logger import logging

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    logging.info("first try sucessful yessss")
    return "first try sucessful"

if __name__ == "__main__":
    app.run(debug=True)