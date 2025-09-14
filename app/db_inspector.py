import sqlalchemy
from sqlalchemy import create_engine, text
import logging
import re
import time
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Generic Helper Functions ---

def get_engine(db_uri: str) -> sqlalchemy.engine.Engine:
    """Creates and tests a SQLAlchemy engine."""
    try:
        engine = create_engine(db_uri, pool_pre_ping=True)
        with engine.connect() as connection:
            logger.info("Database connection successful.")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

def apply_ddl(write_db_uri: str, ddl_statement: str) -> Tuple[bool, str]:
    """Applies a DDL statement to the database, for any dialect."""
    try:
        write_engine = get_engine(write_db_uri)
        with write_engine.connect() as connection:
            # Use a transaction for DDL if the backend supports it.
            # For MySQL, some DDL causes an implicit commit.
            trans = connection.begin()
            try:
                connection.execute(text(ddl_statement))
                trans.commit()
                logger.info(f"Successfully applied DDL: {ddl_statement}")
                return True, "Success."
            except Exception:
                trans.rollback()
                raise
    except Exception as e:
        logger.error(f"Failed to apply DDL: {e}")
        return False, f"Failed to apply fix: {e}"

# --- Abstract Base Class for Database Inspection ---

class DatabaseInspector(ABC):
    """Abstract interface for database-specific inspection operations."""
    def __init__(self, engine: sqlalchemy.engine.Engine):
        self.engine = engine
        self.dialect = engine.dialect.name

    @abstractmethod
    def get_slow_queries(self) -> Dict[str, Any]:
        """Get top N slow queries from the database."""
        pass

    @abstractmethod
    def get_schema_details(self) -> str:
        """Get CREATE statements for tables, views, and indexes."""
        pass

    @abstractmethod
    def get_query_plan(self, query: str) -> Optional[Dict[str, Any]]:
        """Get the estimated query plan (EXPLAIN) as a JSON object."""
        pass

    @abstractmethod
    def get_query_plan_and_execution_time(self, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        """Get the actual query plan and execution time (EXPLAIN ANALYZE)."""
        pass

    @abstractmethod
    def simulate_ddl(self, ddl_statement: str, original_query: str) -> Optional[Dict[str, Any]]:
        """Safely simulate a DDL statement's impact on a query's performance."""
        pass

    def verify_queries_match(self, original_query: str, optimized_query: str) -> Tuple[bool, List[Dict], List[Dict], Optional[str]]:
        """Generic method to execute and compare results of two queries."""
        # This method is dialect-agnostic and can be defined in the base class.
        original_results: List[Dict] = []
        optimized_results: List[Dict] = []
        error_message: Optional[str] = None
        
        if not optimized_query or original_query.strip() == optimized_query.strip():
            optimized_query = original_query
            
        original_query_limited = self._clean_and_limit_query(original_query)
        optimized_query_limited = self._clean_and_limit_query(optimized_query)

        try:
            with self.engine.connect() as connection:
                try:
                    res_orig = connection.execute(text(original_query_limited))
                    original_results = [dict(row._mapping) for row in res_orig.fetchall()]
                except Exception as e:
                    error_message = f"Error executing original query: {e}"
                    return False, [], [], error_message

                if original_query_limited != optimized_query_limited:
                    try:
                        res_opt = connection.execute(text(optimized_query_limited))
                        optimized_results = [dict(row._mapping) for row in res_opt.fetchall()]
                    except Exception as e:
                        error_message = f"Error executing optimized query: {e}"
                        return False, original_results, [], error_message
                else:
                    optimized_results = original_results

            normalized_orig = self._normalize_results(original_results)
            normalized_opt = self._normalize_results(optimized_results)
            
            match = normalized_orig == normalized_opt
            if not match and original_results and optimized_results:
                 if set(original_results[0].keys()) != set(optimized_results[0].keys()):
                    error_message = "Result columns do not match."

        except Exception as e:
            error_message = f"An unexpected error occurred during verification: {e}"
            return False, original_results, optimized_results, error_message

        return match, original_results, optimized_results, error_message

    def _clean_and_limit_query(self, query: str, limit: int = 20) -> str:
        query = query.strip().rstrip(';')
        if not re.search(r'\blimit\s+\d+\s*$', query, re.IGNORECASE):
            query = f"{query} LIMIT {limit}"
        return query

    def _normalize_results(self, results: List[Dict]) -> List[Tuple]:
        if not results:
            return []
        normalized = [tuple(sorted(row.items())) for row in results]
        return sorted(normalized)

# --- PostgreSQL Implementation ---

class PostgresInspector(DatabaseInspector):
    """Contains all the original logic specific to PostgreSQL."""
    def get_slow_queries(self) -> Dict[str, Any]:
        query = text("SELECT query, calls, total_exec_time, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;")
        results = []
        source = "pg_stat_statements"
        try:
            with self.engine.connect() as connection:
                result_proxy = connection.execute(query).fetchall()
                if not result_proxy:
                    logger.warning("pg_stat_statements is empty or has been reset.")
                    return {"source": source, "queries": []}
                for row in result_proxy:
                    results.append({'query': row[0], 'calls': row[1], 'mean_time_ms': row[3]})
                return {"source": source, "queries": results}
        except sqlalchemy.exc.ProgrammingError:
            logger.warning("pg_stat_statements extension not found or not enabled.")
            return {"source": source, "queries": []}
        except Exception as e:
            logger.error(f"Error fetching from pg_stat_statements: {e}")
            raise

    def get_schema_details(self) -> str:
        schema_info = ""
        query_tables = text("""
            SELECT 'CREATE TABLE ' || table_name || ' (' || string_agg(column_name || ' ' || data_type, ', ') || ');' as create_statement
            FROM information_schema.columns WHERE table_schema = 'public' GROUP BY table_name;
        """)
        query_views = text("SELECT 'CREATE VIEW ' || table_name || ' AS ' || view_definition || ';' as create_statement FROM information_schema.views WHERE table_schema = 'public';")
        query_indexes = text("SELECT indexdef as create_statement FROM pg_indexes WHERE schemaname = 'public';")
        try:
            with self.engine.connect() as connection:
                # --- FIX IS APPLIED IN THE FOLLOWING 3 LOOPS ---
                for stmt in connection.execute(query_tables).scalars():
                    if stmt:  # Check if stmt is not None
                        schema_info += stmt + "\n\n"
                for stmt in connection.execute(query_views).scalars():
                    if stmt:  # Check if stmt is not None
                        schema_info += stmt + "\n\n"
                for stmt in connection.execute(query_indexes).scalars():
                    if stmt:  # Check if stmt is not None
                        schema_info += stmt + ";\n"
            return schema_info
        except Exception as e:
            logger.error(f"Error fetching PostgreSQL schema details: {e}", exc_info=True)
            raise

    def get_query_plan(self, query: str) -> Optional[Dict[str, Any]]:
        explain_query = text(f"EXPLAIN (FORMAT JSON) {query}")
        try:
            with self.engine.connect() as connection:
                result = connection.execute(explain_query).scalar_one()
            return result[0].get("Plan") if result else None
        except Exception as e:
            logger.error(f"Error getting PostgreSQL query plan for '{query}': {e}")
            return {"error": str(e)}

    def get_query_plan_and_execution_time(self, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        explain_query = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
        try:
            with self.engine.connect() as connection:
                result_json = connection.execute(explain_query).scalar_one()
                plan = result_json[0].get("Plan")
                execution_time = result_json[0].get("Execution Time")
                return plan, execution_time
        except Exception as e:
            logger.error(f"Error getting analyzed PostgreSQL query plan for '{query}': {e}")
            raise

    def simulate_ddl(self, ddl_statement: str, original_query: str) -> Optional[Dict[str, Any]]:
        try:
            with self.engine.connect() as connection:
                with connection.begin() as transaction:
                    connection.execute(text(ddl_statement))
                    explain_q = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {original_query}")
                    result_json = connection.execute(explain_q).scalar_one()
                    root = result_json[0]
                    plan_obj = root.get("Plan")
                    exec_time = root.get("Execution Time")
                    transaction.rollback()
                    return {"plan": plan_obj, "execution_time_ms": exec_time}
        except Exception as e:
            logger.error(f"Error during PostgreSQL DDL simulation: {e}")
            raise

# --- MySQL Implementation ---

class MySqlInspector(DatabaseInspector):
    """Implements the inspector interface for MySQL."""
    def get_slow_queries(self) -> Dict[str, Any]:
        query = text("""
            SELECT 
                digest_text AS query,
                count_star AS calls,
                (sum_timer_wait / 1000000000) / count_star AS mean_time_ms
            FROM performance_schema.events_statements_summary_by_digest
            WHERE digest_text IS NOT NULL
            ORDER BY sum_timer_wait DESC
            LIMIT 10;
        """)
        results = []
        source = "Performance Schema"
        try:
            with self.engine.connect() as connection:
                # Check if performance schema is enabled
                check_stmt = text("SELECT @@performance_schema;")
                if connection.execute(check_stmt).scalar_one() == 0:
                    logger.warning("MySQL Performance Schema is not enabled.")
                    return {"source": source, "queries": []}
                
                result_proxy = connection.execute(query).fetchall()
                if not result_proxy:
                    logger.warning("MySQL Performance Schema has no data.")
                    return {"source": source, "queries": []}
                
                for row in result_proxy:
                     results.append({'query': row[0], 'calls': row[1], 'mean_time_ms': row[2]})
                return {"source": source, "queries": results}
        except Exception as e:
            logger.error(f"Error fetching from MySQL Performance Schema: {e}")
            # Don't raise, just return empty as it might be a permissions issue.
            return {"source": source, "queries": []}

    def get_schema_details(self) -> str:
        schema_info = ""
        try:
            with self.engine.connect() as connection:
                tables = connection.execute(text("SHOW TABLES;")).scalars().fetchall()
                for table_name in tables:
                    create_stmt = connection.execute(text(f"SHOW CREATE TABLE `{table_name}`;")).fetchone()
                    if create_stmt and create_stmt[1]:
                        schema_info += create_stmt[1] + ";\n\n"
                
                views = connection.execute(text("SHOW FULL TABLES WHERE table_type = 'VIEW';")).fetchall()
                for view_row in views:
                    view_name = view_row[0]
                    create_stmt = connection.execute(text(f"SHOW CREATE VIEW `{view_name}`;")).fetchone()
                    if create_stmt and create_stmt[1]:
                        schema_info += create_stmt[1] + ";\n\n"
            return schema_info
        except Exception as e:
            logger.error(f"Error fetching MySQL schema details: {e}", exc_info=True)
            raise

    def get_query_plan(self, query: str) -> Optional[Dict[str, Any]]:
        explain_query = text(f"EXPLAIN FORMAT=JSON {query}")
        try:
            with self.engine.connect() as connection:
                result = connection.execute(explain_query).scalar_one()
                # MySQL's JSON output is a single object, not a list
            return json.loads(result) if result else None
        except Exception as e:
            logger.error(f"Error getting MySQL query plan for '{query}': {e}")
            return {"error": str(e)}

    def get_query_plan_and_execution_time(self, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        # MySQL EXPLAIN doesn't execute, so we get the plan first, then time the execution.
        plan = self.get_query_plan(query)
        execution_time = None
        try:
            with self.engine.connect() as connection:
                start_time = time.perf_counter()
                connection.execute(text(query)).fetchall() # Fetch all to ensure completion
                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to ms
        except Exception as e:
            logger.error(f"Error executing query for timing: {e}")
            raise
        return plan, execution_time

    def simulate_ddl(self, ddl_statement: str, original_query: str) -> Optional[Dict[str, Any]]:
        logger.warning("DDL simulation with automatic rollback is not supported for MySQL due to implicit commits. Skipping.")
        return None

# --- Factory Function ---

def get_inspector(db_uri: str) -> DatabaseInspector:
    """
    Factory function that returns the correct DatabaseInspector instance
    based on the database URI.
    """
    engine = get_engine(db_uri)
    dialect = engine.dialect.name
    
    if dialect == 'postgresql':
        return PostgresInspector(engine)
    elif dialect == 'mysql':
        return MySqlInspector(engine)
    else:
        raise NotImplementedError(f"Database dialect '{dialect}' is not supported.")