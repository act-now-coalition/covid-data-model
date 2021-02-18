import flask


def create_app():
    """Construct core Flask application with embedded Dash app."""
    # Roughly copied from https://hackersandslackers.com/plotly-dash-with-flask/
    app = flask.Flask(__name__, instance_relative_config=False)
    with app.app_context():
        # Import and initialize Dash application
        from dash_app import dashboard

        app = dashboard.init(app)

        return app
