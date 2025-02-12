'''
Step 5: write to database
Input:
- step4 w2r_canonicalized_results.json
Output:
- to Neo4j
'''

from neo4j import GraphDatabase
from utils import *
import tqdm


class WasteToResourceDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_waste_to_resource(self, w2r_data):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_link, w2r_data)

    @staticmethod
    def _create_and_link(tx, w2r_data):
        for data in tqdm.tqdm(w2r_data):
            query = (
                "MERGE (waste:Waste {name: $waste}) "
                "MERGE (resource:Resource {name: $resource}) "
                "MERGE (waste)-[r:CONVERTED_INTO]->(resource) "
                "SET r.process = $process, r.reference = $reference "
                "RETURN waste, resource, r"
            )
            result = tx.run(query, waste=data['waste'], resource=data['transformed_resource'],
                            process=data['transforming_process'], reference=data['reference'])
            # print(result.single())


def main():
    # Example Usage
    uri = "neo4j://localhost:7687"  # Modify with your actual URI
    user = "neo4j"  # Modify with your actual user
    password = "w2rkg_final_v2"  # Modify with your actual password

    db = WasteToResourceDB(uri, user, password)

    ## AN EXAMPLE TO TEST
    # w2r_data = [
    #     {
    #         "waste": "date pits",
    #         "transforming_process": [
    #             "pyrolysis",
    #             "co-pyrolysis"
    #         ],
    #         "transformed_resource": "char",
    #         "reference": "10.1016/j.psep.2024.04.101"
    #     }
    # ]

    w2r_data = read_json("result_all/after_fusion_v2/thre08_complete/fused_triples_aggregated.json")

    # write to database
    db.create_waste_to_resource(w2r_data)
    db.close()


if __name__ == "__main__":
    main()
