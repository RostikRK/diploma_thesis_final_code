from DigiKeyClient import DigiKeyClient
from neo4j import GraphDatabase
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from helpers.Neo4j_operator import Neo4jOperations, process_result


# Set up start
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

DIGI_CLIENT_ID = ""
DIGI_CLIENT_SECRET = ""
# Set up end

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

neo4j_ops = Neo4jOperations(driver)

# Base QUERY_SPEC
QUERY_SPEC = {
    "Keywords": "",
    "Limit": 50,
    "Offset": 0,
    "FilterOptionsRequest": {
        "CategoryFilter": [{"Id": "774"}],
        "ManufacturerFilter": [],
        "StatusFilter": [
            {"Id": "0"}, {"Id": "2"}, {"Id": "4"}, {"Id": "7"}
        ],
        "PackagingFilter": [],
        "MarketPlaceFilter": "NoFilter",
        "MinimumQuantityAvailable": 0,
        "ParameterFilterRequest": {
            "CategoryFilter": {"Id": "774"},
            "ParameterFilters": []
        }
    },
    "SortOptions": {
        "Field": "None",
        "SortOrder": "Ascending"
    }
}

# Specific Filter Data
manufacturers_group1 = [
    1528, 1265, 600061, 1450, 544, 122, 734, 313, 1282, 708,
    5369, 428, 3247, 714, 722, 602238, 3052, 600434, 1970
]
value_ids_group1 = [
    '720365', '720430', '120546', '168307', '720427', '720410', '720346',
    '720385', '261668', '720333', '720367', '720431', '720351', '720375',
    '720395', '640383', '827273', '640384', '827273', '640385', '720393',
    '640386', '720425', '640387', '720364', '720429', '720360', '720381',
    '720428'
]
packaging_17_value_ids = [
    '720241', '720245', '719887', '719889', '719894', '720278', '720286',
    '720220', '720228', '720230', '720182', '720190', '720192', '719858',
    '108137', '719867', '720196', '720203', '720204', '720159', '720168',
    '720169', '719833', '719841', '720259', '720264', '719878'
]
packaging_1_value_ids = [
    '720410', '720346', '720385', '261668', '720333', '720367', '720431',
    '720351', '720375', '640383', '1'
]
packaging_other_ids = [3, 2, 243, 6]
packaging_other_value_ids = [
    '720410', '720346', '720385', '261668', '720333', '720367', '720431',
    '720351', '720375'
]
manufacturers_group2 = [
    1982, 5107, 706, 709, 712, 3217, 5297, 5625, 1853, 711,
    5110, 726, 1092, 150
]
value_ids_group2 = [
    '720365', '720430', '120546', '168307', '720427', '720410', '720346',
    '720385', '261668', '720333', '720367', '720431', '720351', '720375',
    '720395', '824896', '640383', '824881', '640384', '824883', '640385',
    '824880', '640386', '827298', '824885', '640387', '720364', '720429',
    '824895', '720360', '720381', '720428', '730010'
]
manufacturers_group3 = [
    557, 730, 1727, 600046, 488, 20, 1695, 2156, 585, 600057, 728, 725, 723,
    425, 1855, 1984, 502239, 2120, 1274, 1568, 602241, 497, 1052, 296, 601860,
    716, 718, 720, 256, 732
]

value_ids_group3 = [
    '720365', '720430', '120546', '168307', '720427', '720410', '720346', '720348', '720385', '720387', '261668',
    '720333', '720337', '720367', '720431', '720351', '720375', '720395', '640383', '640384', '640385', '720393',
    '640386', '827298', '720425', '720411', '640387', '720343', '720364', '720382', '720390', '720429', '720438',
    '802520', '720360', '720381', '720428', '730010', '827278', '1'
]



def generate_configs():
    """Generates the combinations of filter configs. """
    configs = []
    manuf1_filter = []
    for manufacturer in manufacturers_group1:
        manuf1_filter.append({"Id": str(manufacturer)})
    for value_id in value_ids_group1:
        config = {
            "ManufacturerFilter": manuf1_filter,
            "PackagingFilter": [],
            "ParameterFilter": [
                {"ParameterId": 961, "FilterValues": [{"Id": '347758'}]},
                {"ParameterId": 142, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_17_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "448"}],
            "PackagingFilter": [{"Id": "17"}],
            "ParameterFilter": [
                {"ParameterId": 961, "FilterValues": [{"Id": '347758'}]},
                {"ParameterId": 2689, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_1_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "448"}],
            "PackagingFilter": [{"Id": "1"}],
            "ParameterFilter": [
                {"ParameterId": 961, "FilterValues": [{"Id": '347758'}]},
                {"ParameterId": 142, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    packag1_filter = []
    for packaging_id in packaging_other_ids:
        packag1_filter.append({"Id": str(packaging_id)})
    for value_id in packaging_other_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "448"}],
            "PackagingFilter": packag1_filter,
            "ParameterFilter": [
                {"ParameterId": 961, "FilterValues": [{"Id": '347758'}]},
                {"ParameterId": 142, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    manuf2_filter = []
    for manufacturer in manufacturers_group2:
        manuf2_filter.append({"Id": str(manufacturer)})
    for value_id in value_ids_group2:
        config = {
            "ManufacturerFilter": manuf2_filter,
            "PackagingFilter": [],
            "ParameterFilter": [
                {"ParameterId": 961, "FilterValues": [{"Id": '347758'}]},
                {"ParameterId": 142, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    manuf3_filter = []
    for manufacturer in manufacturers_group3:
        manuf3_filter.append({"Id": str(manufacturer)})
    for value_id in value_ids_group3:
        config = {
            "ManufacturerFilter": manuf3_filter,
            "PackagingFilter": [],
            "ParameterFilter": [
                {"ParameterId": 961, "FilterValues": [{"Id": '347758'}]},
                {"ParameterId": 142, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    return configs

def fetch_products_for_config(config):
    """Fetches and stores information about products using one config."""
    digikey = DigiKeyClient(DIGI_CLIENT_ID, DIGI_CLIENT_SECRET)
    print(config)
    QUERY_SPEC["FilterOptionsRequest"]["ManufacturerFilter"] = config["ManufacturerFilter"]
    QUERY_SPEC["FilterOptionsRequest"]["PackagingFilter"] = config["PackagingFilter"]
    QUERY_SPEC["FilterOptionsRequest"]["ParameterFilterRequest"]["ParameterFilters"] = config["ParameterFilter"]
    offset = 0
    limit = 50
    QUERY_SPEC["Offset"] = offset
    QUERY_SPEC["Limit"] = limit
    results = digikey.search_products(QUERY_SPEC)
    if results and "FilterOptions" in results and "TopCategories" in results["FilterOptions"]:
        total_products = results["FilterOptions"]["TopCategories"][0]["Category"]["ProductCount"]
        process_result(results, neo4j_ops)
        while offset + limit < total_products:
            offset += limit
            QUERY_SPEC["Offset"] = offset
            results = digikey.search_products(QUERY_SPEC)
            if results:
                # print(config)
                print(offset + limit, total_products)
                process_result(results, neo4j_ops)
            else:
                print(f"No more results for config: {config}")
                break
    else:
        print(f"No products found for config: {config}")


if __name__ == "__main__":
    all_configs = generate_configs()
    print(len(all_configs))
    for ind, config in enumerate(all_configs):
        print("Config Num: ", ind+1)
        fetch_products_for_config(config)
