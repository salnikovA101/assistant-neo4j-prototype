from neo4j import AsyncGraphDatabase

_driver = None

def init_driver(uri, user, password):
    global _driver
    if _driver is None:
        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    return _driver

def get_driver():
    global _driver
    if _driver is None:
        raise RuntimeError("Neo4j driver is not initialized. Call init_driver first.")
    return _driver

async def close_driver():
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
