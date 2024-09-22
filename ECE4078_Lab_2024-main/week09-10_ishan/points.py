import json
import os

def extract_coordinates(input_filepath, output_filepath):
    """
    Extracts the x and y coordinates of markers from a JSON-like string in a file and 
    writes them to a text file in a specific format.

    Parameters:
        input_filepath (str): The path to the input file containing the JSON-like string.
        output_filepath (str): The path to the output file to write the coordinates to.
    """
    try:
        # Read the JSON-like string from the input file
        with open(input_filepath, "r") as f:
            data_str = f.read()

        # Parse the string into a Python dictionary
        data = json.loads(data_str)

        # Extract taglist and map
        taglist = data["taglist"]
        map_coordinates = data["map"]

        # Create a dictionary to store the coordinates of each marker
        markers_coordinates = {}

        # Pair each marker with its x and y coordinates
        for i, tag in enumerate(taglist):
            marker_name = f"aruco{tag}_0"
            markers_coordinates[marker_name] = {"x": map_coordinates[0][i], "y": map_coordinates[1][i]}

        # Convert the markers_coordinates dictionary into a JSON string and save it to a new text file
        with open(output_filepath, "w") as f:
            json.dump(markers_coordinates, f, indent=4)
                
        print(f"Coordinates have been written to {output_filepath}")

    except json.JSONDecodeError:
        print("Error: Input string is not valid JSON.")
    except KeyError as e:
        print(f"Error: Key {e} not found in data.")
    except FileNotFoundError:
        print(f"Error: File {input_filepath} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def append_fruit_coordinates(targets_filepath, output_filepath):
    """
    Appends the fruit coordinates from targets.txt to coordinates.txt.

    Parameters:
        targets_filepath (str): The path to the input file containing the fruit coordinates.
        output_filepath (str): The path to the output file to append the coordinates to.
    """
    try:
        # Load the existing marker coordinates
        with open(output_filepath, "r") as f:
            markers_coordinates = json.load(f)

        # Load the fruit coordinates
        with open(targets_filepath, "r") as f:
            fruit_coordinates = json.load(f)

        # Append the fruit coordinates to the marker coordinates
        markers_coordinates.update(fruit_coordinates)

        # Write the updated coordinates back to the output file
        with open(output_filepath, "w") as f:
            json.dump(markers_coordinates, f, indent=4)

        print(f"Fruit coordinates have been appended to {output_filepath}")

    except json.JSONDecodeError:
        print("Error: Input string is not valid JSON.")
    except FileNotFoundError:
        print(f"Error: File {targets_filepath} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Define file paths
input_filepath = os.path.join("lab_output", "slam.txt")
output_filepath = os.path.join("lab_output", "points.txt")
targets_filepath = os.path.join("lab_output", "targets.txt")

# Extract coordinates and write them to the output file
extract_coordinates(input_filepath, output_filepath)

# Append fruit coordinates to the output file
append_fruit_coordinates(targets_filepath, output_filepath)