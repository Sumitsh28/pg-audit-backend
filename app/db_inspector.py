import sqlalchemy
from sqlalchemy import create_engine, text
import logging

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
                schema_info += table[0] + "\n\n"
            
            views = connection.execute(query_views).fetchall()
            for view in views:
                schema_info += view[0] + "\n\n"
            
            indexes = connection.execute(query_indexes).fetchall()
            for index in indexes:
                schema_info += index[0] + ";\n"
        return schema_info
    except Exception as e:
        logger.error(f"Error fetching schema details: {e}")
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