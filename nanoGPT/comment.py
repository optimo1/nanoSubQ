file_path = "your_file.txt"

# Read the file and add a '#' to the start of every line
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

commented_lines = ["# " + line for line in lines]

# Overwrite the original file with the commented code
with open(file_path, "w", encoding="utf-8") as file:
    file.writelines(commented_lines)

print(f"Successfully commented out everything in {file_path}")
