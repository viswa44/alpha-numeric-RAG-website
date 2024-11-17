import os
import camelot
import pandas as pd


def extract_tables_from_pdfs(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over PDF files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, file_name)
            print(f"Processing: {pdf_path}")

            try:
                # Extract tables using Camelot
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # Use 'lattice' for grid-based tables

                if tables.n > 0:
                    print(f"Found {tables.n} table(s) in {file_name}.")

                    for i, table in enumerate(tables):
                        # Prepare output JSON path
                        output_json = os.path.join(
                            output_folder, f"{os.path.splitext(file_name)[0]}_table_{i + 1}.json"
                        )

                        # Convert the table to a DataFrame and save as JSON
                        table_df = table.df  # Extracted table as a DataFrame
                        table_df.to_json(output_json, orient='records', indent=4)
                        print(f"Table {i + 1} saved to: {output_json}")
                else:
                    print(f"No tables found in {file_name}.")
            
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


# Example usage
# input_folder = "ipdata"  # Replace with the path to your input folder containing PDFs
# output_folder = "opdata"  # Replace with the path to your desired output folder
# extract_tables_from_pdfs(input_folder, output_folder)
