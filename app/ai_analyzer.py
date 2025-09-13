import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

import re

from .models import AISuggestion, CostSlayerInfo

load_dotenv()
logger = logging.getLogger(__name__)

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None
    
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
    Constructs a strict, example-backed prompt that forces a single minified JSON response.
    """
    # small defensive extraction of top-level numeric fields for model context
    top_fields = {}
    if isinstance(query_plan, dict):
        for k in ("Total Cost", "Plan Rows", "Actual Rows", "Node Type"):
            if k in query_plan:
                top_fields[k] = query_plan[k]
    # build prompt with explicit JSON shape + tiny example
    return f"""
You are an expert PostgreSQL performance analyst. Output MUST be a single minified JSON object with EXACT keys:
"rewritten_query", "new_index_suggestion", "explanation"

Context:
SCHEMA:
{schema_details}

SLOW QUERY:
{query}

PLAN_TOP_FIELDS:
{json.dumps(top_fields, ensure_ascii=False)}

Rules:
1) If you recommend an index, it MUST exactly match: CREATE INDEX index_name ON table_name (col1, col2);
2) If the query involves a VIEW, target the underlying base table(s).
3) Prefer index suggestions only when they will reduce planner Total Cost; otherwise provide a rewritten_query.
4) No markdown. No extra keys. No commentary outside the JSON object.

Example (this EXACT shape is required):
{{"rewritten_query":"SELECT ...","new_index_suggestion":"CREATE INDEX idx_foo_bar ON foo(bar);","explanation":"index on foo.bar to speed join"}}
"""

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
            model="gpt-3.5-mini",
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
    if not client:
        raise ConnectionError("OpenAI client not initialized. Check your API key.")
    prompt = construct_prompt(schema_details, query, query_plan)
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        try:
            parsed = validate_and_parse_ai_response(content)
        except ValueError:
            # retry once with slightly higher temperature (give model another chance)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content": prompt + "\nRetry: output EXACT JSON object only."}],
                response_format={"type":"json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            parsed = validate_and_parse_ai_response(content)

        # Final sanity: limit explanation length
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