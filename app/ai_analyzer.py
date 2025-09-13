import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

from .models import AISuggestion, CostSlayerInfo

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

def construct_prompt(schema_details: str, query: str, query_plan: dict) -> str:
    # This function remains the same as before
    return f"""
    You are an expert PostgreSQL performance analyst. Your task is to analyze a slow query and provide an optimal solution.

    Here is the database context:
    --- SCHEMA ---
    {schema_details}
    --- END SCHEMA ---

    Here is the slow query and its execution plan:
    --- SLOW QUERY ---
    {query}
    --- END SLOW QUERY ---

    --- EXECUTION PLAN (JSON) ---
    {json.dumps(query_plan, indent=2)}
    --- END EXECUTION PLAN ---

    Analysis Task:
    1.  Identify the primary bottleneck in the execution plan. Look for "Seq Scan" (Sequential Scan) on large tables where an index could be used, inefficient joins, or other common issues.
    2.  Based on the bottleneck, determine the best fix. This will either be creating a new, optimal index OR rewriting the query. Prioritize creating an index if it's the clear solution.
    3.  Provide a brief, clear explanation of WHY this fix is necessary and HOW it solves the problem.
    4.  Format your response as a single, minified JSON object with NO markdown formatting (e.g., no ```json).

    The JSON object must have these exact keys:
    - "rewritten_query": A string containing the rewritten SQL query, or null if you are suggesting an index.
    - "new_index_suggestion": A string containing the full `CREATE INDEX` statement, or null if you are rewriting the query.
    - "explanation": A concise string explaining the problem and your solution.
    """

# --- NEW FUNCTION ---
def generate_benchmark_queries(schema_details: str) -> List[str]:
    """
    Calls OpenAI to generate a set of safe, read-only benchmark queries.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized.")

    logger.info("Generating AI benchmark queries...")
    prompt = f"""
    You are an expert PostgreSQL DBA tasked with creating a performance health check.
    Based on the following database schema, generate a list of 3 to 5 diverse, read-only (SELECT only)
    SQL queries that are likely to reveal performance bottlenecks.

    - Include queries with JOINs on unindexed or partially indexed columns.
    - Include queries with aggregate functions (COUNT, SUM, AVG) and GROUP BY.
    - Include queries with WHERE clauses using non-indexed columns or LIKE operators.
    - DO NOT generate trivial queries like 'SELECT * FROM table LIMIT 10'. The queries should be realistic.
    - CRITICAL: Ensure all queries are READ-ONLY (SELECT statements only) and syntactically correct for PostgreSQL.

    --- SCHEMA ---
    {schema_details}
    --- END SCHEMA ---

    Format your response as a single, minified JSON object with one key: "queries", which is an array of SQL query strings.
    Example Response: {{"queries": ["SELECT ...", "SELECT ..."]}}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        content = response.choices[0].message.content
        result_json = json.loads(content)
        logger.info(f"Generated {len(result_json.get('queries', []))} benchmark queries.")
        return result_json.get("queries", [])
    except Exception as e:
        logger.error(f"Error generating benchmark queries from OpenAI: {e}")
        return []

# --- END NEW FUNCTION ---

def get_ai_suggestion(schema_details: str, query: str, query_plan: dict) -> AISuggestion:
    # This function remains the same as before
    if not client:
        raise ConnectionError("OpenAI client not initialized. Check your API key.")

    prompt = construct_prompt(schema_details, query, query_plan)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        
        content = response.choices[0].message.content
        ai_json = json.loads(content)
        
        return AISuggestion(**ai_json)

    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return AISuggestion(
            rewritten_query=None,
            new_index_suggestion=None,
            explanation=f"Could not get AI suggestion due to an error: {e}"
        )

# --- MODIFIED FUNCTION ---
def calculate_cost_slayer(cost_before: float, cost_after: float, calls: int) -> CostSlayerInfo:
    """
    Calculates cost info based on query plan costs.
    Now more direct, using before and after costs.
    """
    # Placeholder logic: Assume 1 cost unit = $0.0001
    cost_per_query = cost_before * 0.0001
    daily_cost = cost_per_query * calls
    
    potential_savings = 0.0
    if cost_before > 0:
        potential_savings = 1 - (cost_after / cost_before)

    return CostSlayerInfo(
        estimated_daily_cost=round(daily_cost, 2),
        potential_savings_percentage=round(potential_savings * 100, 1),
        cost_before=round(cost_before, 2),
        cost_after=round(cost_after, 2)
    )
# --- END MODIFIED FUNCTION ---