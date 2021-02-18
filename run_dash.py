from dash_app import create_app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, use_debugger=True, use_reloader=True)
