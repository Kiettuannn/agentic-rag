import os
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

from configs.setting import load_config
from src.ingestion.loader import load_relationships
from src.retrieval.graph import normalize_relation

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def ingest_to_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    config = load_config()
    relationships = load_relationships(config)

    def create_graph_data(tx, source_id, target_id, rel_type):
        query = (
            "MERGE (s:Document {doc_id: $source_id}) "
            "MERGE (t:Document {doc_id: $target_id}) "
            f"MERGE (s)-[:{rel_type}]->(t)"
        )
        tx.run(query, source_id=source_id, target_id=target_id)

    print(f"Pushing {len(relationships)} relations into Neo4j...")
    with driver.session() as session:
        count = 0
        for row in relationships:
            source = row.get("doc_id", "")
            target = row.get("other_doc_id", "")
            raw_rel = row.get("relationship", "")

            if not source or not target:
                continue

            rel_type, reserve = normalize_relation(raw_rel)
            if reserve:
                source, target = target, source

            safe_rel_type = rel_type.replace(" ", "_").upper()
            session.execute_write(create_graph_data, source, target, safe_rel_type)
            count += 1
            if count % 1000 == 0:
                print(f"Pushed {count} relations into Neo4j...")
    print("Done ingestion")
    driver.close()

if __name__ == "__main__":
    ingest_to_neo4j()


