from flask import Flask, render_template
import matplotlib.pyplot as plt
import numpy as np

# Generate some example data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the Flask app
app = Flask(__name__)

# Define the route to display the plot
@app.route("/")
def plot():
    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Example Plot")

    # Save the plot to a PNG file
    fig.savefig("static/plot.png")

    # Render the HTML template with the plot
    return render_template("plot.html")

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
