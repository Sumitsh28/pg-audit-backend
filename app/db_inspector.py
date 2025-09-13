import sqlalchemy
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_engine(db_uri: str):
    """Creates and returns a SQLAlchemy engine."""
    try:
        engine = create_engine(db_uri, pool_pre_ping=True)
        # Test connection
        with engine.connect() as connection:
            logger.info("Database connection successful.")
        return engine
    except Exception as e:
        logger.error(f"Failed to connect to the database: {e}")
        raise

def get_pg_stat_statements(engine):
    """
    Fetches query performance data from the pg_stat_statements view.
    Returns a list of query data or None if the view is not available or empty.
    """
    query = text("""
        SELECT
            query,
            calls,
            total_exec_time,
            mean_exec_time
        FROM pg_stat_statements
        ORDER BY mean_exec_time DESC
        LIMIT 10;
    """)
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
    Introspects the database to get CREATE TABLE and CREATE INDEX statements.
    This provides crucial context for the AI.
    """
    schema_info = ""
    query_tables = text("""
        SELECT 'CREATE TABLE ' || table_name || ' (' ||
               string_agg(column_name || ' ' || data_type, ', ') ||
               ');' as create_statement
        FROM information_schema.columns
        WHERE table_schema = 'public'
        GROUP BY table_name;
    """)
    query_indexes = text("""
        SELECT indexdef as create_statement
        FROM pg_indexes
        WHERE schemaname = 'public';
    """)
    try:
        with engine.connect() as connection:
            tables = connection.execute(query_tables).fetchall()
            for table in tables:
                schema_info += table[0] + "\n\n"
            
            indexes = connection.execute(query_indexes).fetchall()
            for index in indexes:
                schema_info += index[0] + ";\n"
        return schema_info
    except Exception as e:
        logger.error(f"Error fetching schema details: {e}")
        raise

def get_query_plan(engine, query: str):
    """
    Runs EXPLAIN (FORMAT JSON) on a given query to get its execution plan.
    Does NOT execute the query.
    """
    explain_query = text(f"EXPLAIN (FORMAT JSON) {query}")
    try:
        with engine.connect() as connection:
            result = connection.execute(explain_query).scalar_one()
            return result
    except Exception as e:
        logger.error(f"Error getting query plan for '{query}': {e}")
        return {"error": str(e)}

# --- NEW FUNCTION ---
def get_query_plan_and_execution_time(engine, query: str):
    """
    Runs EXPLAIN (ANALYZE, FORMAT JSON) to get the plan and actual execution time.
    This is for benchmark/user-submitted queries.
    Returns a tuple: (plan, execution_time_ms)
    """
    explain_query = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
    try:
        with engine.connect() as connection:
            # The result is a list with a single JSON object
            result_json = connection.execute(explain_query).scalar_one()
            
            # Extract the plan and the execution time
            plan = result_json[0].get("Plan")
            execution_time = result_json[0].get("Execution Time")
            
            return plan, execution_time
    except Exception as e:
        logger.error(f"Error getting analyzed query plan for '{query}': {e}")
        # Propagate the error to be handled by the main endpoint
        raise e
# --- END NEW FUNCTION ---

def simulate_ddl(engine, ddl_statement: str, original_query: str):
    """
    Safely simulates a DDL change by running it in a transaction
    that is immediately rolled back. It returns the new query plan.
    """
    try:
        with engine.connect() as connection:
            with connection.begin() as transaction:
                connection.execute(text(ddl_statement))
                
                # Use the non-analyzing get_query_plan for simulation
                plan_after = get_query_plan(connection, original_query)
                
                transaction.rollback() # CRITICAL: This undoes the DDL
                logger.info("DDL simulation successful and rolled back.")
                return plan_after
    except Exception as e:
        logger.error(f"Error during DDL simulation: {e}")
        raise

def apply_ddl(write_db_uri: str, ddl_statement: str):
    """
    Connects with write permissions and applies a DDL statement.
    THIS IS A DESTRUCTIVE OPERATION.
    """
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