import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

import re
from typing import List, Dict,  Optional
from pydantic import BaseModel, Field

from .models import AISuggestion, CostSlayerInfo, Problem, OptimizationResult, ChatMessage , ProviderCostSavings

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

    max_min_matches = re.findall(r"\(\s*SELECT\s+(MAX|MIN)\([^\)]+\).*?\)", query, flags=re.IGNORECASE | re.DOTALL)
    if max_min_matches:
        patterns["max_min_subqueries"].extend(max_min_matches)

    in_matches = re.findall(r"IN\s*\(\s*SELECT[^\)]+\)", query, flags=re.IGNORECASE | re.DOTALL)
    if in_matches:
        patterns["nested_in_subqueries"].extend(in_matches)

    or_matches = re.findall(r"\bWHERE\b.*\bOR\b.*", query, flags=re.IGNORECASE | re.DOTALL)
    if or_matches:
        patterns["or_conditions"].extend(or_matches)

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
    s = stmt.strip().rstrip(';')  
    pattern = r'(?i)^CREATE\s+INDEX\s+[A-Za-z0-9_]+\s+ON\s+[A-Za-z0-9_]+\s*\(\s*[A-Za-z0-9_]+(?:\s*,\s*[A-Za-z0-9_]+)*\s*\)\s*$'
    return re.match(pattern, s) is not None


def validate_and_parse_ai_response(raw: str) -> dict:
    """
    MODIFIED: Now validates the new keys for time and data scan estimates.
    """
    required_keys = (
        "rewritten_query", "new_index_suggestion", "explanation",
        "estimated_execution_time_after_ms",
        "estimated_data_scanned_before_mb",
        "estimated_data_scanned_after_mb"
    )

    if not raw or not isinstance(raw, str):
        raise ValueError("Empty or invalid raw response")
    
    s = raw.strip()
    
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r'(\{(?:[^{}]|(?R))*\})', s, flags=re.DOTALL)
        if not m:
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

    for k in required_keys:
        if k not in obj:
            raise ValueError(f"Missing required key in AI response: {k}")
    
    if obj["rewritten_query"] is not None and not isinstance(obj["rewritten_query"], str):
        raise ValueError("Key rewritten_query must be a string or null")
    if obj["new_index_suggestion"] is not None and not isinstance(obj["new_index_suggestion"], str):
        raise ValueError("Key new_index_suggestion must be a string or null")
    if obj["explanation"] is not None and not isinstance(obj["explanation"], str):
        raise ValueError("Key explanation must be a string or null")
    
    for k in ("estimated_execution_time_after_ms", "estimated_data_scanned_before_mb", "estimated_data_scanned_after_mb"):
        if obj[k] is not None and not isinstance(obj[k], (int, float)):
             raise ValueError(f"Key {k} must be a number or null")

    for k in ("rewritten_query", "new_index_suggestion"):
        if obj.get(k) is not None and obj[k].strip() == "":
            obj[k] = None

    return obj

def construct_prompt(schema_details: str, query: str, query_plan: dict, execution_time_ms: float) -> str:
    """
    CORRECTED VERSION: Combines the original detailed rules with the new estimation requirements.
    """
    top_fields = {}
    if isinstance(query_plan, dict):
        for k in ("Total Cost", "Plan Rows", "Actual Rows", "Node Type", "Filter", "Join Type"):
            if k in query_plan:
                top_fields[k] = query_plan[k]

    prompt = f"""
You are an expert PostgreSQL performance analyst. Optimize queries for speed, cost, and scalability.
Output MUST be a single minified JSON object with EXACT keys: 
"rewritten_query", "new_index_suggestion", "explanation", "estimated_execution_time_after_ms", "estimated_data_scanned_before_mb", "estimated_data_scanned_after_mb"

CONTEXT:
SCHEMA:
{schema_details}

SLOW QUERY:
{query}

INITIAL PERFORMANCE:
- Execution Time: {execution_time_ms:.2f} ms

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
8. **Estimate the new execution time in milliseconds** after your optimization and set it in "estimated_execution_time_after_ms". Be realistic based on the plan changes.
9. **Estimate the data scanned in megabytes (MB)** before and after your optimization. Set these values in "estimated_data_scanned_before_mb" and "estimated_data_scanned_after_mb".
10. Output JSON ONLY, no extra text, no markdown, no commentary.

EXAMPLE:
{{
"rewritten_query":"WITH top_films AS (SELECT film_id FROM film_revenue ORDER BY total_revenue DESC LIMIT 10), ranked_customers AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY film_id ORDER BY rental_count DESC) AS rn FROM film_customers) SELECT fr.title, fr.total_revenue, rc.customer_id, rc.rental_count FROM film_revenue fr JOIN top_films tf ON fr.film_id = tf.film_id JOIN ranked_customers rc ON fr.film_id = rc.film_id WHERE rc.rn = 1;",
"new_index_suggestion":"CREATE INDEX idx_film_customers_film_rental ON film_customers(film_id, rental_count);",
"explanation":"Rewrote per-row MAX() subquery using ROW_NUMBER() window function to avoid repeated scans; added covering index to speed join.",
"estimated_execution_time_after_ms": 55.5,
"estimated_data_scanned_before_mb": 1500.0,
"estimated_data_scanned_after_mb": 120.0
}}
"""
    return prompt



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

def get_ai_suggestion(schema_details: str, query: str, query_plan: dict, execution_time_ms: float) -> AISuggestion:
    """
    MODIFIED: Now accepts execution_time_ms to pass to the prompt.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized. Check your API key.")
    
    prompt = construct_prompt(schema_details, query, query_plan, execution_time_ms)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Using a stronger model can improve estimation quality
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        parsed = validate_and_parse_ai_response(content)
        return AISuggestion(**parsed)

    except Exception as e:
        logger.error(f"Error in get_ai_suggestion: {e} -- raw_ai: {locals().get('content')}")
        return AISuggestion(
            rewritten_query=None, new_index_suggestion=None, explanation=f"AI failed: {e}",
            estimated_execution_time_after_ms=None,
            estimated_data_scanned_before_mb=None,
            estimated_data_scanned_after_mb=None
        )


def calculate_cost_slayer(scanned_before_mb: float, scanned_after_mb: float) -> CostSlayerInfo:
    """
    REWRITTEN: Calculates potential cost savings PER CALL for different cloud providers.
    """
    if scanned_before_mb is None or scanned_after_mb is None or scanned_before_mb <= scanned_after_mb:
        return CostSlayerInfo(potential_savings=[])

    MB_PER_TB = 1024 * 1024
    USD_TO_INR = 88.27

    providers = {
        "Google BigQuery": 6.25,
        "AWS Athena": 5.00
    }
    
    data_saved_mb = scanned_before_mb - scanned_after_mb
    data_saved_tb = data_saved_mb / MB_PER_TB
    
    savings_list = []
    for name, rate_usd_per_tb in providers.items():
        # Calculate per-call savings, not daily
        savings_usd_per_call = data_saved_tb * rate_usd_per_tb
        savings_inr_per_call = savings_usd_per_call * USD_TO_INR
        
        savings_list.append(ProviderCostSavings(
            provider_name=name,
            # Use the new field name and DO NOT round the value
            potential_savings_inr_per_call=savings_inr_per_call 
        ))
        
    return CostSlayerInfo(potential_savings=savings_list)
    
def get_rag_chat_response(
    schema_details: str,
    query_context,  
    optimization_context: Optional[object],  
    chat_history: List[object],  
    user_question: str
) -> str:
    """
    Generates a response for a chatbot using a RAG pattern.
    The context is built from the provided schema, query, plan, and optimization.
    """
    if not client:
        raise ConnectionError("OpenAI client not initialized.")

    parts: List[str] = []
    parts.append("--- CONTEXT DOCUMENT START ---\n")
    parts.append(
        "You are a helpful AI assistant for a PostgreSQL optimization tool. "
        "Your task is to answer questions about a specific SQL query and its performance analysis.\n\n"
        "You MUST strictly base your answers on the context provided below.\n"
        "If the user's question cannot be answered using this context, state that you do not have enough information from the provided context.\n"
        "Do not answer questions unrelated to this specific query. Do not suggest optimizations not mentioned in the context.\n"
    )

    parts.append("**Database Schema:**\n")
    parts.append(f"{schema_details}\n")

    parts.append("**Query Under Discussion:**\n")
    parts.append("```sql\n")
    parts.append(f"{query_context.query}\n")
    parts.append("```\n\n")

    parts.append("Performance Analysis (Before Optimization):\n\n")
    parts.append(f"Execution Time: {query_context.execution_time_ms:.2f} ms\n")
    parts.append(f"Number of Calls: {query_context.calls}\n\n")
    parts.append("Query Plan (Before):\n")
    
    try:
        parts.append(json.dumps(query_context.query_plan_before, indent=2) + "\n")
    except Exception:
        parts.append("No query plan available or failed to serialize.\n")

    if optimization_context:
        ai_sugg = getattr(optimization_context, "ai_suggestion", None)
        cost_slayer = getattr(optimization_context, "cost_slayer", None)
        parts.append("\nAI-Powered Optimization Suggestion:\n\n")
        if ai_sugg:
            parts.append(f"Rewritten Query: {ai_sugg.rewritten_query or 'No rewrite suggested.'}\n\n")
            parts.append(f"New Index Suggestion: {ai_sugg.new_index_suggestion or 'No index suggested.'}\n\n")
            parts.append(f"Explanation for Suggestion: {ai_sugg.explanation or 'No explanation provided.'}\n\n")
        else:
            parts.append("- No AI suggestion object available.\n\n")

        if cost_slayer:
            parts.append("Estimated Performance (After Optimization):\n\n")
            parts.append(f"Cost (Before): {getattr(cost_slayer, 'cost_before', 'N/A')}\n")
            parts.append(f"Cost (After): {getattr(cost_slayer, 'cost_after', 'N/A')}\n")
        else:
            parts.append("- No cost slayer information available.\n")

        est_time = getattr(optimization_context, "estimated_execution_time_after_ms", None)
        parts.append(f"Estimated Execution Time: {(est_time or 0):.2f} ms\n")
    else:
        parts.append("\nAI-Powered Optimization Suggestion:\n")
        parts.append("- The 'Optimize' function has not been run for this query yet.\n")

    parts.append("\n--- CONTEXT DOCUMENT END ---\n")

    context_doc = "".join(parts)

    messages_for_api = [{"role": "system", "content": context_doc}]
    for msg in chat_history:
        messages_for_api.append({"role": msg.role, "content": msg.content})
    messages_for_api.append({"role": "user", "content": user_question})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_for_api,
            temperature=0.1,
        )
       
        try:
            return response.choices[0].message.content
        except Exception:
           
            try:
                return response.choices[0].text
            except Exception:
                logger.error("Unexpected response format from OpenAI: %s", response)
                return "Sorry, I received an unexpected response format from the AI."
    except Exception as e:
        logger.error(f"Error getting RAG chat response from OpenAI: {e}")
        return "Sorry, I encountered an error while processing your question."