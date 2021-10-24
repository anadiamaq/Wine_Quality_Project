from app import app
from flask import Flask
import os
from fetch import *

if __name__ == "__main__":

    app.run("0.0.0.0", port=os.getenv('PORT'), debug=True)
