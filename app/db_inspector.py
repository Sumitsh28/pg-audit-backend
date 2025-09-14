import sqlalchemy
from sqlalchemy import create_engine, text
import logging

import re

from typing import List, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_engine(db_uri: str):
  
    try:
        engine = create_engine(db_uri, pool_pre_ping=True)
        with engine.connect() as connection:
            logger.info("Database connection successful.")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

def get_pg_stat_statements(engine):
    
    query = text("SELECT query, calls, total_exec_time, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;")
    try:
        with engine.connect() as connection:
            result = connection.execute(query).fetchall()
            if not result:
                logger.warning("pg_stat_statements is empty or has been reset.")
                return None
            return result
    except sqlalchemy.exc.ProgrammingError:
        logger.warning("pg_stat_statements extension not found or not enabled.")
        return None
    except Exception as e:
        logger.error(f"Error fetching from pg_stat_statements: {e}")
        raise

def get_schema_details(engine):
    """
    Introspects the database to get CREATE TABLE, CREATE VIEW, and CREATE INDEX statements.
    """
    schema_info = ""
   
    query_tables = text("""
        SELECT 'CREATE TABLE ' || table_name || ' (' || string_agg(column_name || ' ' || data_type, ', ') || ');' as create_statement
        FROM information_schema.columns WHERE table_schema = 'public' GROUP BY table_name;
    """)
    
    query_views = text("""
        SELECT 'CREATE VIEW ' || table_name || ' AS ' || view_definition || ';' as create_statement
        FROM information_schema.views WHERE table_schema = 'public';
    """)
    
    query_indexes = text("SELECT indexdef as create_statement FROM pg_indexes WHERE schemaname = 'public';")
    
    try:
        with engine.connect() as connection:
            
            tables = connection.execute(query_tables).fetchall()
            for table in tables:
                if table and table[0]:
                    schema_info += table[0] + "\n\n"
            
            views = connection.execute(query_views).fetchall()
            for view in views:
                if view and view[0]:
                    schema_info += view[0] + "\n\n"
            
            indexes = connection.execute(query_indexes).fetchall()
            for index in indexes:
                if index and index[0]:
                    schema_info += index[0] + ";\n"
        return schema_info
    except Exception as e:
        logger.error(f"Error fetching schema details: {e}", exc_info=True)
        raise

def get_query_plan(db_connection_or_engine, query: str):
    """
    Runs EXPLAIN (FORMAT JSON) and returns the plan OBJECT, not the list.
    """
    explain_query = text(f"EXPLAIN (FORMAT JSON) {query}")
    try:
        if isinstance(db_connection_or_engine, sqlalchemy.engine.base.Engine):
            with db_connection_or_engine.connect() as connection:
                result = connection.execute(explain_query).scalar_one()
        else: 
            result = db_connection_or_engine.execute(explain_query).scalar_one()
        
        return result[0].get("Plan") if result else None
    except Exception as e:
        logger.error(f"Error getting query plan for '{query}': {e}")
        return {"error": str(e)}

def get_query_plan_and_execution_time(engine, query: str):
   
    explain_query = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
    try:
        with engine.connect() as connection:
            result_json = connection.execute(explain_query).scalar_one()
            plan = result_json[0].get("Plan")
            execution_time = result_json[0].get("Execution Time")
            return plan, execution_time
    except Exception as e:
        logger.error(f"Error getting analyzed query plan for '{query}': {e}")
        raise e

def simulate_ddl(engine, ddl_statement: str, original_query: str):
    """
    Apply DDL in a transaction, run EXPLAIN (ANALYZE, FORMAT JSON) on original_query,
    capture plan + Execution Time + Total Cost, then rollback.
    Returns dict: {"plan": plan_obj, "execution_time_ms": exec_time, "total_cost": total_cost}
    """
    try:
        with engine.connect() as connection:
            with connection.begin() as transaction:
                
                connection.execute(text(ddl_statement))
               
                explain_q = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {original_query}")
                result_json = connection.execute(explain_q).scalar_one()
                plan_obj = None
                exec_time = None
                total_cost = None
                if isinstance(result_json, (list, tuple)) and len(result_json) > 0:
                    root = result_json[0]
                    plan_obj = root.get("Plan")
                    exec_time = root.get("Execution Time")
                    if plan_obj:
                        total_cost = plan_obj.get("Total Cost") or root.get("Total Cost")
             
                transaction.rollback()
                return {"plan": plan_obj, "execution_time_ms": exec_time, "total_cost": total_cost}
    except Exception as e:
        logger.error(f"Error during DDL simulation: {e}")
        raise

def apply_ddl(write_db_uri: str, ddl_statement: str):
    write_engine = get_engine(write_db_uri)
    try:
        with write_engine.connect() as connection:
            connection.execution_options(isolation_level="AUTOCOMMIT")
            connection.execute(text(ddl_statement))
        logger.info(f"Successfully applied DDL: {ddl_statement}")
        return True, "Success."
    except Exception as e:
        logger.error(f"Failed to apply DDL: {e}")
        return False, f"Failed to apply fix: {e}"
    
def _clean_and_limit_query(query: str, limit: int = 20) -> str:
    """A helper to safely remove trailing semicolons and add a LIMIT clause."""
    query = query.strip().rstrip(';')
    # Use regex to avoid adding a second LIMIT clause
    if not re.search(r'\blimit\s+\d+\s*$', query, re.IGNORECASE):
        query = f"{query} LIMIT {limit}"
    return query

def _normalize_results(results: List[Dict]) -> List[Tuple]:
    """Converts a list of dicts for order-independent comparison."""
    if not results:
        return []
    # Creates a sorted list of sorted tuples of items, making comparison deterministic
    normalized = [tuple(sorted(row.items())) for row in results]
    return sorted(normalized)

def verify_queries_match(engine, original_query: str, optimized_query: str) -> Tuple[bool, List[Dict], List[Dict], str]:
    """
    Executes two queries with a LIMIT 20, converts results to a list of dicts,
    and checks if the contents are identical, ignoring row and column order.
    """
    original_results: List[Dict] = []
    optimized_results: List[Dict] = []
    error_message: str = None
    
    # If there's no rewritten query, the results are inherently the same.
    # We set them as equal so we only need to execute one for the preview.
    if not optimized_query or original_query.strip() == optimized_query.strip():
        optimized_query = original_query
        
    original_query_limited = _clean_and_limit_query(original_query)
    optimized_query_limited = _clean_and_limit_query(optimized_query)

    try:
        with engine.connect() as connection:
            # Execute original query
            try:
                res_orig = connection.execute(text(original_query_limited))
                keys_orig = res_orig.keys()
                original_results = [dict(zip(keys_orig, row)) for row in res_orig.fetchall()]
            except Exception as e:
                error_message = f"Error executing original query: {e}"
                return False, [], [], error_message

            # Execute optimized query only if it's different
            if original_query_limited != optimized_query_limited:
                try:
                    res_opt = connection.execute(text(optimized_query_limited))
                    keys_opt = res_opt.keys()
                    optimized_results = [dict(zip(keys_opt, row)) for row in res_opt.fetchall()]
                except Exception as e:
                    error_message = f"Error executing optimized query: {e}"
                    return False, original_results, [], error_message
            else:
                optimized_results = original_results

        # Compare results
        normalized_orig = _normalize_results(original_results)
        normalized_opt = _normalize_results(optimized_results)
        
        if normalized_orig == normalized_opt:
            match = True
        else:
            match = False
            if original_results and optimized_results:
                if list(original_results[0].keys()) != list(optimized_results[0].keys()):
                    error_message = "Result columns do not match in name or order."

    except Exception as e:
        logger.error(f"An unexpected error occurred in verify_queries_match: {e}", exc_info=True)
        error_message = f"An unexpected error occurred: {e}"
        return False, original_results, optimized_results, error_message

    return match, original_results, optimized_results, error_message