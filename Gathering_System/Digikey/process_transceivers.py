import sys
import os
from DigiKeyClient import DigiKeyClient
from neo4j import GraphDatabase
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
        "CategoryFilter": [{"Id": "710"}],
        "ManufacturerFilter": [],
        "StatusFilter": [
            {"Id": "0"}, {"Id": "2"}, {"Id": "4"}, {"Id": "7"}
        ],
        "PackagingFilter": [],
        "MarketPlaceFilter": "NoFilter",
        "MinimumQuantityAvailable": 0,
        "ParameterFilterRequest": {
            "CategoryFilter": {"Id": "710"},
            "ParameterFilters": []
        }
    },
    "SortOptions": {
        "Field": "None",
        "SortOrder": "Ascending"
    }
}

# Filter Data
manufacturers_group1 = [
    5503, 600061, 1016, 544, 4430, 766, 505
]
value_ids_group1 = [
    '293994', '294288', '294620', '294628', '294755', '294918', '294926', '294958', '294976', '295340',
    '73277', '73532', '73980', '73983', '87043', '87149', '87405', '814445', '97190', '97321', '97322',
    '97350', '107870', '108077', '108442', '108530', '108548', '108607', '108613', '108645', '108652',
    '108693', '116061', '116402', '116473', '116474', '142844', '143602', '143699', '143701', '143723',
    '143756', '154992', '155044', '155045', '155093', '155117', '155156', '165703', '165723', '165749',
    '165779', '443932', '165794', '165804', '165835', '165842', '188441', '188446', '188725', '197335',
    '200190', '200207', '200210', '214766', '214768', '215261', '215264', '220341', '220500', '225889',
    '226034', '226196', '261751', '716253', '68820', '405438'
]

packaging_1_17_value_ids = [
    '261579', '78844', '79184', '84023', '85626', '140341', '150582', '150456', '151187', '153413',
    '156882', '198041', '222746', '225168', '236693', '120547', '55328', '168308', '131794', '202165',
    '227911', '248675', '73641', '87237', '108196', '143185', '158812', '184168', '214848', '217906',
    '238189', '68941', '140492', '259921', '1'
]

packaging_6_250_value_ids = [
    '261579', '78844', '79184', '84022', '84023', '85626', '105820',
    '140341', '150582', '150456', '151187', '153413', '156882', '198041',
    '222746', '236693', '120547', '55328', '168308', '131794', '202165',
    '227911', '248675', '73641', '87237', '108196', '143185', '158812', '184168',
    '214848', '217906', '238189', '68941', '259921', '456757', '1'
]
packaging_other_ids = [61, 3, 2, 243]
packaging_other_value_ids = [
    '311745', '325925', '341180', '342399', '358006', '359803', '361812', '365059',
    '369718', '378370', '398950', '398992', '399041', '399047', '399058', '399079',
    '409933', '418344', '418468', '418589'
]
manufacturers_group2 = [
    516, 428, 31, 5272, 600039, 600055, 600026, 1245, 448, 1902, 600010, 600060, 2521, 2208, 413, 576, 150
]
value_ids_group2 = [
    '242918', '242919', '242941', '242948', '212434', '212437',
    '212478', '212479', '212507', '212508', '212510', '212514',
    '212617', '212621', '139411', '38976', '38985', '39003', '39008', '1'
]
manufacturers_group3 = [
    2725, 600066, 14, 2129, 390, 568, 488, 660, 20, 1695, 846, 600016, 863, 1568, 497, 5536, 4518, 3351
]

value_ids_group3 = [
    '73461', '493835', '142966', '142967', '142968', '199032', '289198', '78843', '78844', '84022',
    '84023', '85626', '86548', '86549', '140341', '147760', '150582', '633949', '150456',
    '709916', '156882', '158049', '213226', '222746', '236692', '236693', '120544', '120547',
    '168308', '123297', '131794', '248675', '294413', '73641', '73652', '87237', '103180', '143185',
    '158812', '184168', '214848', '238189', '68941', '101636', '140492', '156984', '213311', '225194', '1'
]

packaging_6_17_value_ids = [
    '1', '325925', '341180', '345107', '350548', '350612', '357980', '357985', '357999',
    '358000', '358001', '358006', '358016', '365041', '365059', '369718', '369761', '374442',
    '375628', '398950', '399047', '399058', '418598', '422442'
]

packaging_1_value_ids = [
    '1', '325925', '341180', '345107', '429199', '350612', '356906', '357985', '358000', '358001',
    '358006', '358016', '359803', '365041', '365059', '369718', '369761', '375015', '375628',
    '398950', '398992', '399047', '399058', '409753', '418344', '418468', '418598'
]

packaging_3_2_243_value_ids = [
    '1', '325925', '341180', '345107', '429199', '350548', '350612', '356906', '357980',
    '357985', '358000', '358001', '358006', '358016', '359803', '365041', '365059', '369718',
    '369761', '374442', '375015', '375628', '398950', '398992', '399041', '399047', '399058',
    '409753', '418344', '418468', '418598'
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
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 1291, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_1_17_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "175"}],
            "PackagingFilter": [{"Id": "1"}, {"Id": "17"}],
            "ParameterFilter": [
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 448, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_6_250_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "175"}],
            "PackagingFilter": [{"Id": "6"}, {"Id": "250"}],
            "ParameterFilter": [
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 448, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    packag1_filter = []
    for packaging_id in packaging_other_ids:
        packag1_filter.append({"Id": str(packaging_id)})
    for value_id in packaging_other_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "175"}],
            "PackagingFilter": packag1_filter,
            "ParameterFilter": [
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 578, "FilterValues": [{"Id": value_id}]}
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
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 252, "FilterValues": [{"Id": value_id}]}
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
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 448, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_6_17_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "296"}],
            "PackagingFilter": [{"Id": "6"}, {"Id": "17"}],
            "ParameterFilter": [
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 578, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_1_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "296"}],
            "PackagingFilter": [{"Id": "1"}],
            "ParameterFilter": [
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 578, "FilterValues": [{"Id": value_id}]}
            ]
        }
        configs.append(config)
    for value_id in packaging_3_2_243_value_ids:
        config = {
            "ManufacturerFilter": [{"Id": "296"}],
            "PackagingFilter": [{"Id": "3"}, {"Id": "2"}, {"Id": "243"}],
            "ParameterFilter": [
                {"ParameterId": 183, "FilterValues": [{"Id": '415534'}]},
                {"ParameterId": 578, "FilterValues": [{"Id": value_id}]}
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
        print("Config Num: ", ind + 1)
        fetch_products_for_config(config)
