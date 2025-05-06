import csv
import os

# --- Configuration ---
# <<< --- IMPORTANT: Set the correct path to YOUR dataset --- >>>
BASE_DATA_PATH = os.environ.get('/Users/karlnuyda/Desktop/Fantasy-Premier-League/data') # Use your actual path
SEASON = '2024-25'
PROBLEM_FILE = os.path.join(BASE_DATA_PATH, SEASON, 'gws', 'merged_gw.csv')
PROBLEM_LINE_NUMBER = 14180 # The line number reported in the error (1-based)
# --- End Configuration ---

def analyze_csv_line(file_path, target_line_num):
    """
    Reads the header and a specific target line from a CSV to compare fields.
    """
    print(f"Analyzing file: {file_path}")
    print(f"Looking for structure difference on line: {target_line_num}\n")

    header_fields = []
    problem_line_fields = []
    found_problem_line = False

    try:
        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csvfile:
            # Use csv.reader to handle commas within quoted fields correctly
            reader = csv.reader(csvfile)

            for i, row_fields in enumerate(reader):
                current_line_num = i + 1 # csv.reader is 0-indexed, errors are 1-based

                # Capture header
                if current_line_num == 1:
                    header_fields = row_fields
                    print(f"Header Row (Line 1) - Expected Fields: {len(header_fields)}")
                    # print("Header Fields:", header_fields) # Uncomment to see all header names

                # Capture problem line
                if current_line_num == target_line_num:
                    problem_line_fields = row_fields
                    found_problem_line = True
                    print(f"\nProblem Row (Line {target_line_num}) - Actual Fields Found: {len(problem_line_fields)}")
                    # print("Problem Fields:", problem_line_fields) # Uncomment to see raw problem fields
                    break # Stop after finding the problem line

            if not header_fields:
                 print("Error: Could not read header row.")
                 return

            if not found_problem_line:
                print(f"Error: Did not reach line number {target_line_num}. File might be shorter.")
                return

            # --- Comparison ---
            print("\n--- Field Comparison ---")
            max_fields = max(len(header_fields), len(problem_line_fields))

            for i in range(max_fields):
                header_val = header_fields[i] if i < len(header_fields) else "---NO HEADER---"
                problem_val = problem_line_fields[i] if i < len(problem_line_fields) else "---NO VALUE---"

                # Highlight potential discrepancies
                prefix = "  "
                if i >= len(header_fields) or i >= len(problem_line_fields):
                     prefix = ">>" # Indicates mismatch in length
                elif header_val != problem_val and i < len(header_fields):
                     # This comparison isn't perfect as it's data vs header name,
                     # but helps visualize alignment.
                     pass # Simple presence/absence is key here

                print(f"{prefix} Field #{i+1}: Header='{header_val}' | Problem Line Value='{problem_val}'")

            if len(problem_line_fields) > len(header_fields):
                print(f"\n>> Found {len(problem_line_fields) - len(header_fields)} extra field(s) on the problem line.")
            elif len(problem_line_fields) < len(header_fields):
                 print(f"\n>> Found {len(header_fields) - len(problem_line_fields)} fewer field(s) on the problem line.")


    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

# --- Main execution block ---
if __name__ == "__main__":
    if not os.path.isfile(PROBLEM_FILE):
         print(f"Error: Problem file path not found: {PROBLEM_FILE}", file=sys.stderr)
         print("Please update the BASE_DATA_PATH variable in the script.", file=sys.stderr)
    else:
        analyze_csv_line(PROBLEM_FILE, PROBLEM_LINE_NUMBER)