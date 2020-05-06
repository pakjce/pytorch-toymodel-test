from flask import Flask
from os.path import dirname, join


def create_app():
    app = Flask(
        __name__,
        template_folder=join(dirname(__file__), 'templates')
    )

    from .index import bp_index
    from .inference import bp_inference
    app.register_blueprint(bp_index)
    app.register_blueprint(bp_inference)

    return app
