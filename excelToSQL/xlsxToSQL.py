import pandas as pd

def excel_to_sql_inserts(excel_file, table_name, output_file):
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Start generating SQL code
    sql_code = ""

    # Generate INSERT statements
    for _, row in df.iterrows():
        columns = ', '.join(df.columns)
        values = ', '.join([f"'{str(x).replace('\'', '\'\'')}'" for x in row])
        sql_code += f"INSERT INTO {table_name} ({columns}) VALUES ({values});\n"

    # Write to output SQL file
    with open(output_file, 'w') as file:
        file.write(sql_code)

    print(f"SQL insert statements have been successfully written to {output_file}")

# Example usage
excel_file = r'C:\Users\User\Documents\GitHub\Python-Project\excelToSQL\Member_Data_v2.xlsx'
table_name = 'member'
output_file = 'inserts.sql'

excel_to_sql_inserts(excel_file, table_name, output_file)
