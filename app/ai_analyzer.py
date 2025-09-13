import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

import re
from typing import List, Dict

from .models import AISuggestion, CostSlayerInfo

load_dotenv()
logger = logging.getLogger(__name__)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

def detect_query_patterns(query: str) -> Dict[str, List[str]]:
    """
    Detect common SQL bottleneck patterns:
    - MAX/MIN subqueries
    - Nested IN subqueries
    - OR conditions
    - Aggregations without filters
    Returns a dict summarizing the patterns found.
    """
    patterns = {
        "max_min_subqueries": [],
        "nested_in_subqueries": [],
        "or_conditions": [],
        "aggregations_no_filter": [],
    }

    # Detect MAX()/MIN() in WHERE or SELECT as a subquery
    max_min_matches = re.findall(r"\(\s*SELECT\s+(MAX|MIN)\([^\)]+\).*?\)", query, flags=re.IGNORECASE | re.DOTALL)
    if max_min_matches:
        patterns["max_min_subqueries"].extend(max_min_matches)

    # Detect IN (SELECT ...) subqueries
    in_matches = re.findall(r"IN\s*\(\s*SELECT[^\)]+\)", query, flags=re.IGNORECASE | re.DOTALL)
    if in_matches:
        patterns["nested_in_subqueries"].extend(in_matches)

    # Detect OR conditions in WHERE clauses
    or_matches = re.findall(r"\bWHERE\b.*\bOR\b.*", query, flags=re.IGNORECASE | re.DOTALL)
    if or_matches:
        patterns["or_conditions"].extend(or_matches)

    # Detect aggregations without filters
    agg_matches = re.findall(r"SELECT\s+.*\b(COUNT|SUM|AVG|MIN|MAX)\(", query, flags=re.IGNORECASE)
    if agg_matches and "WHERE" not in query.upper():
        patterns["aggregations_no_filter"].extend(agg_matches)

    return patterns
    
def is_safe_create_index(stmt: str) -> bool:
    """
    Very strict regex to allow only simple CREATE INDEX statements:
    CREATE INDEX index_name ON table_name (col1, col2);
    """
    if not stmt or not isinstance(stmt, str):
        return False
    s = stmt.strip().rstrip(';')  # allow trailing semicolon or not
    pattern = r'(?i)^CREATE\s+INDEX\s+[A-Za-z0-9_]+\s+ON\s+[A-Za-z0-9_]+\s*\(\s*[A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)*\s*\)\s*$'
    return re.match(pattern, s) is not None


def validate_and_parse_ai_response(raw: str, required_keys=("rewritten_query","new_index_suggestion","explanation")) -> dict:
    """
    Try to extract a single JSON object from raw model text, parse it, and validate required keys.
    Returns the parsed dict or raises ValueError.
    """
    if not raw or not isinstance(raw, str):
        raise ValueError("Empty or invalid raw response")

    # Quick trim
    s = raw.strip()

    # 1) If it is valid JSON at top-level, try parsing
    try:
        obj = json.loads(s)
    except Exception:
        # 2) Otherwise try to extract the first {...} block
        m = re.search(r'(\{(?:[^{}]|(?R))*\})', s, flags=re.DOTALL)
        if not m:
            # fallback: try to find a substring starting with { and ending with }
            m2 = re.search(r'(\{.*\})', s, flags=re.DOTALL)
            if not m2:
                raise ValueError("Could not find JSON object in AI response")
            raw_json = m2.group(1)
        else:
            raw_json = m.group(1)
        try:
            obj = json.loads(raw_json)
        except Exception as e:
            raise ValueError(f"Failed to parse extracted JSON: {e} -- raw snippet: {raw_json[:200]}")

    # 3) Validate required keys exist and types are sane
    for k in required_keys:
        if k not in obj:
            raise ValueError(f"Missing required key in AI response: {k}")
        if obj[k] is not None and not isinstance(obj[k], str):
            raise ValueError(f"Key {k} must be a string or null")

    # 4) Normalize empty strings to null for rewrites/index suggestions
    for k in ("rewritten_query","new_index_suggestion"):
        if obj.get(k) is not None and obj[k].strip() == "":
            obj[k] = None

    return obj

def construct_prompt(schema_details: str, query: str, query_plan: dict) -> str:
    """
    Constructs an advanced PostgreSQL optimization prompt.
    Includes multiple rewrite patterns, index suggestions, and cost-aware guidance.
    AI must output a single minified JSON object: rewritten_query, new_index_suggestion, explanation.
    """
    top_fields = {}
    if isinstance(query_plan, dict):
        for k in ("Total Cost", "Plan Rows", "Actual Rows", "Node Type", "Filter", "Join Type"):
            if k in query_plan:
                top_fields[k] = query_plan[k]

    prompt = f"""
You are an expert PostgreSQL performance analyst. Optimize queries for speed, cost, and scalability.
Output MUST be a single minified JSON object with EXACT keys: 
"rewritten_query", "new_index_suggestion", "explanation"

CONTEXT:
SCHEMA:
{schema_details}

SLOW QUERY:
{query}

QUERY_PLAN:
{json.dumps(query_plan, ensure_ascii=False)}

RULES:
1. Only generate syntactically correct PostgreSQL SQL.
2. Always check for and optimize these patterns:
   a) Per-row MAX/MIN subqueries → use ROW_NUMBER() or RANK() window functions.
   b) OR conditions across multiple columns → consider UNION ALL.
   c) Nested IN (SELECT ...) → rewrite as JOIN when possible.
   d) Aggregations on large tables → push filters before aggregation.
   e) JOINs on non-indexed columns → suggest covering index if it reduces Total Cost.
3. Prefer index creation ONLY if it reduces Total Cost or speeds up joins/filters significantly. Use: CREATE INDEX index_name ON table_name(col1, col2);
4. Push down WHERE filters early, flatten unnecessary nested queries.
5. If no index is beneficial, set "new_index_suggestion": null.
6. If query cannot be further optimized, set "rewritten_query": null.
7. Explanation must briefly justify why the rewrite or index improves performance.
8. Output JSON ONLY, no extra text, no markdown, no commentary.
9. Format JSON EXACTLY: 
   {{"rewritten_query":"...","new_index_suggestion":"...","explanation":"..."}}

EXAMPLES:
Original query:
SELECT fr.title, fr.total_revenue, fc.customer_id, fc.rental_count
FROM film_revenue fr
JOIN film_customers fc ON fr.film_id = fc.film_id
WHERE fr.film_id IN (SELECT film_id FROM film_revenue ORDER BY total_revenue DESC LIMIT 10)
  AND fc.rental_count = (SELECT MAX(fc2.rental_count) FROM film_customers fc2 WHERE fc2.film_id = fr.film_id);

Optimized output:
{{
"rewritten_query":"WITH top_films AS (SELECT film_id FROM film_revenue ORDER BY total_revenue DESC LIMIT 10), ranked_customers AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY film_id ORDER BY rental_count DESC) AS rn FROM film_customers) SELECT fr.title, fr.total_revenue, rc.customer_id, rc.rental_count FROM film_revenue fr JOIN top_films tf ON fr.film_id = tf.film_id JOIN ranked_customers rc ON fr.film_id = rc.film_id WHERE rc.rn = 1;",
"new_index_suggestion":"CREATE INDEX idx_film_customers_film_rental ON film_customers(film_id, rental_count);",
"explanation":"Rewrote per-row MAX() subquery using ROW_NUMBER() window function to avoid repeated scans; added covering index to speed join."
}}
"""
    return prompt


# --- (The rest of the file is unchanged from the last fix) ---

def generate_benchmark_queries(schema_details: str) -> List[str]:
    """
    Calls OpenAI to generate a set of safe, read-only benchmark queries.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized.")

    logger.info("Generating AI benchmark queries...")
    prompt = f"""
    You are an expert PostgreSQL DBA tasked with creating a performance health check.
    Based on the following database schema, generate a list of 2 diverse, read-only (SELECT only) SQL queries.

    --- SCHEMA ---
    {schema_details}
    --- END SCHEMA ---

    Rules for query generation:
    1.  Queries must be realistic and designed to find performance bottlenecks (e.g., using JOINs, aggregations, filtering on non-indexed columns).
    2.  **CRITICAL DATA TYPE RULE:** Pay very close attention to the column data types in the schema. When creating `JOIN` conditions (`ON table1.col1 = table2.col2`), you **MUST** ensure the columns being compared have compatible data types (e.g., integer with integer, text with text). Do not generate joins between incompatible types like integer and text.
    3.  **CRITICAL SAFETY RULE:** All queries **MUST** be READ-ONLY (`SELECT` statements only) and syntactically correct for PostgreSQL.
    4.  Format your response as a single, minified JSON object with one key: "queries", which is an array of SQL query strings.
    Example Response: {{"queries": ["SELECT ...", "SELECT ..."]}}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        result_json = json.loads(content)
        logger.info(f"Generated {len(result_json.get('queries', []))} benchmark queries.")
        return result_json.get("queries", [])
    except Exception as e:
        logger.error(f"Error generating benchmark queries from OpenAI: {e}")
        return []

def get_ai_suggestion(schema_details: str, query: str, query_plan: dict) -> AISuggestion:
    """
    Returns an AI-optimized suggestion for a given PostgreSQL query.
    Automatically detects common bottlenecks and augments the prompt.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized. Check your API key.")

    # --- Detect common patterns ---
    patterns = detect_query_patterns(query)

    # --- Construct advanced prompt ---
    prompt = construct_prompt(schema_details, query, query_plan)
    # Include detected patterns for AI context
    if any(patterns.values()):
        prompt += f"\nDETECTED_PATTERNS: {patterns}"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        try:
            parsed = validate_and_parse_ai_response(content)
        except ValueError:
            # Retry once with stricter instructions
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt + "\nRetry: output EXACT JSON object only."}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            parsed = validate_and_parse_ai_response(content)

        # Limit explanation length
        if len(parsed.get("explanation","")) > 300:
            parsed["explanation"] = parsed["explanation"][:297] + "..."

        return AISuggestion(**parsed)

    except Exception as e:
        logger.error(f"Error in get_ai_suggestion: {e} -- raw_ai: {locals().get('content')}")
        return AISuggestion(rewritten_query=None, new_index_suggestion=None, explanation=f"AI failed: {e}")


def calculate_cost_slayer(cost_before: float, cost_after: float, calls: int) -> CostSlayerInfo:
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