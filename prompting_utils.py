import os
import re


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    if not os.path.exists(schema_path):
        return "Schema file not found."
    
    with open(schema_path, 'r') as f:
        schema_content = f.read()
    
    return schema_content


def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # Clean up the response
    response = response.strip()
    
    # Try to find SQL query between SQL code blocks
    sql_pattern = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if sql_pattern:
        return sql_pattern.group(1).strip()
    
    # Try to find SQL query between generic code blocks
    code_pattern = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_pattern:
        return code_pattern.group(1).strip()
    
    # Look for "SQL query:" pattern which is common in our prompt format
    sql_label_pattern = re.search(r'SQL query:\s*(.*?)(?:\n\n|$)', response, re.DOTALL)
    if sql_label_pattern:
        return sql_label_pattern.group(1).strip()
    
    # For T5 models, the output might be just the SQL query without any formatting
    # Check if the response starts with common SQL keywords
    sql_keywords = r'(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)'
    query_pattern = re.search(f'^{sql_keywords}.*?(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if query_pattern:
        return query_pattern.group(0).strip()
    
    # If no direct match, try to find any SQL-like pattern in the text
    query_pattern = re.search(f'{sql_keywords}.*?(?:;|$)', response, re.DOTALL | re.IGNORECASE)
    if query_pattern:
        return query_pattern.group(0).strip()
    
    # If all else fails, return the entire response
    # But first check if it contains any SQL-like fragments
    if re.search(r'(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY)', response, re.IGNORECASE):
        # Try to clean up the response to make it more SQL-like
        # Remove any natural language explanations
        cleaned = re.sub(r'^.*?(SELECT|WITH)', r'\1', response, flags=re.DOTALL | re.IGNORECASE)
        # Remove anything after a terminal semicolon
        cleaned = re.sub(r';.*$', ';', cleaned, flags=re.DOTALL)
        return cleaned.strip()
    
    # Handle the case where the model repeats the schema or prompt
    if "Database schema:" in response:
        # Try to extract anything after the schema that looks like SQL
        after_schema = response.split("Database schema:")[-1]
        after_schema = after_schema.split("Natural language query:")[-1]
        
        # Look for SQL keywords in what remains
        query_pattern = re.search(f'{sql_keywords}.*?(?:;|$)', after_schema, re.DOTALL | re.IGNORECASE)
        if query_pattern:
            return query_pattern.group(0).strip()
    
    return response.strip()


def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\n")
        f.write(f"Error rate: {sum(1 for msg in error_msgs if msg) / len(error_msgs) if error_msgs else 0}\n")
        f.write("Error Messages:\n")
        for i, msg in enumerate(error_msgs):
            if msg:
                f.write(f"Query {i}: {msg}\n")