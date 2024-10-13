import os
from app import app

if __name__ == "__main__":
    os.environ["FLASK_APP"] = "app.py"
    app.run(host="0.0.0.0", port=3000, debug=True)
