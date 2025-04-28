import types
from helpers.attribute_dictionary_handler import fndict
from helpers.attribute_handler_functions import snake

class Neo4jOperations:
    def __init__(self, driver):
        self.driver = driver

    def create_or_update_manufacturer(self, manufacturer_name):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_manufacturer_tx, manufacturer_name)

    def _create_or_update_manufacturer_tx(self, tx, manufacturer_name):
        query = """
        MERGE (c:Manufacturer {name: $name})
        RETURN c
        """
        tx.run(query, name=manufacturer_name)

    def create_or_update_datasheet(self, datasheet_url):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_datasheet_tx, datasheet_url)

    def _create_or_update_datasheet_tx(self, tx, datasheet_url):
        query = """
        MERGE (d:Datasheet {name: $url})
        RETURN d
        """
        tx.run(query, url=datasheet_url)

    def link_product_to_datasheet(self, product_key, datasheet_url, state):
        with self.driver.session() as session:
            session.execute_write(self._link_product_to_datasheet_tx, product_key, datasheet_url, state)

    def _link_product_to_datasheet_tx(self, tx, product_key, datasheet_url, state):
        query = """
        MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
        MATCH (d:Datasheet {name: $datasheet_url})
        MERGE (p)-[r:HAS_DATASHEET]->(d)
        SET r.state = $state
        """
        tx.run(query, product_name=product_key['pn'], manufacturer_name=product_key['manufacturer'],
               datasheet_url=datasheet_url, state=state)

    def create_or_update_product(self, product_data, manufacturer_name, processed):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_product_tx, product_data, manufacturer_name, processed)

    def mark_datasheet_as(self, datasheet_url, state):
        query = """
        MATCH (p:Product)-[r:HAS_DATASHEET]->(d:Datasheet)
        WHERE d.name = $datasheet_url AND r.state<>"Undownloadable"
        SET r.state = $state
        """
        with self.driver.session() as session:
            session.execute_write(lambda tx: tx.run(query, datasheet_url=datasheet_url, state=state))
        print(f"Marked datasheet {datasheet_url} as {state}.")

    def check_category_exists(self, category_name):
        query = """
        MATCH (c:Category {name: $category_name})
        RETURN count(c) as count
        """
        with self.driver.session() as session:
            result = session.execute_read(lambda tx: tx.run(query, category_name=category_name).data())
            return result[0]["count"] > 0

    def _create_or_update_product_tx(self, tx, product_data, manufacturer_name, price_processed):
        # First ensure the manufacturer exists
        query1 = """
        MERGE (m:Manufacturer {name: $manufacturer_name})
        RETURN m
        """

        # Then update the product and link it to its manufacturer
        query2 = """
        MATCH (m:Manufacturer {name: $manufacturer_name})
        MERGE (p:Product {
            name: $name,
            manufacturer: $manufacturer_name
        })
        SET p.description = $description,
            p.price_processed = $price_processed,
            p.status = $status
        MERGE (p)-[:MANUFACTURED_BY]->(m)
        RETURN p
        """

        tx.run(query1, manufacturer_name=manufacturer_name)
        tx.run(query2,
               name=product_data["ManufacturerProductNumber"],
               manufacturer_name=product_data["Manufacturer"]["Name"],
               description=product_data["Description"]["DetailedDescription"],
               price_processed=price_processed,
               status=product_data["ProductStatus"]["Status"])

    def link_product_manufacturer(self, product_name, manufacturer_name):
        with self.driver.session() as session:
            session.execute_write(self._link_product_manufacturer_tx, product_name, manufacturer_name)

    def _link_product_manufacturer_tx(self, tx, product_name, manufacturer_name):
        query = """
        MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
        MATCH (c:Manufacturer {name: $manufacturer_name})
        MERGE (p)-[:MANUFACTURED_BY]->(c)
        """
        tx.run(query, product_name=product_name, manufacturer_name=manufacturer_name)

    def create_or_update_category(self, category_data):
        if category_data["name"] == 'Drivers, Receivers, Transceivers':
            category_data["name"] = "Transceivers"
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_category_tx, category_data)

    def _create_or_update_category_tx(self, tx, category_data):
        query = """
        MERGE (c:Category {name: $name})
        SET c.description = $description
        RETURN c
        """
        tx.run(query,
               name=category_data["name"],
               description=category_data["description"])

    def link_product_category(self, product_key, category_name):
        with self.driver.session() as session:
            session.execute_write(self._link_product_category_tx, product_key, category_name)

    def _link_product_category_tx(self, tx, product_key, category_name):
        query = """
        MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
        MATCH (c:Category {name: $category_name})
        MERGE (p)-[:BELONGS_TO]->(c)
        """
        tx.run(query, product_name=product_key['pn'], manufacturer_name=product_key['manufacturer'],
               category_name=category_name)

    def link_subcategory(self, subcategory_name, parent_category_name):
        with self.driver.session() as session:
            session.execute_write(self._link_subcategory_tx, subcategory_name, parent_category_name)

    def _link_subcategory_tx(self, tx, subcategory_name, parent_category_name):
        query = """
        MATCH (sub:Category {name: $subcategory_name})
        MATCH (parent:Category {name: $parent_category_name})
        MERGE (sub)-[:SUBCATEGORY_OF]->(parent)
        """
        tx.run(query, subcategory_name=subcategory_name, parent_category_name=parent_category_name)

    def create_or_update_attribute(self, attr_data):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_attribute_tx, attr_data)

    def _create_or_update_attribute_tx(self, tx, attr_data):
        query = """
        MERGE (a:Attribute {name: $name})
        SET a.description = $description,
            a.possible_symbols = $possible_symbols
        RETURN a
        """
        tx.run(query,
               name=attr_data["name"],
               description=attr_data["description"],
               possible_symbols=attr_data["possible_symbols"])

    def link_product_attribute(self, product_key, attribute_name, value, test_conditions=""):
        with self.driver.session() as session:
            session.execute_write(self._link_product_attribute_tx, product_key,
                                  attribute_name, value, test_conditions)

    def _link_product_attribute_tx(self, tx, product_key, attribute_name, value, test_conditions=""):
        query = """
        MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
        MATCH (a:Attribute {name: $attribute_name})
        MERGE (p)-[r:HAS_ATTRIBUTE]->(a)
        SET r.value = $value,
            r.test_conditions = $test_conditions
        """
        tx.run(query,
               product_name=product_key['pn'],
               manufacturer_name=product_key['manufacturer'],
               attribute_name=attribute_name,
               test_conditions=test_conditions,
               value=value)

    def create_or_update_package(self, package_data):
        with self.driver.session() as session:
            session.execute_write(self._create_or_update_package_tx, package_data)

    def _create_or_update_package_tx(self, tx, package_data):
        query = """
        MERGE (a:Package {name: $package_name})
        SET a.standardized_package_names = $standardized_package_names
        RETURN a
        """
        tx.run(query,
               package_name=package_data["package_name"],
               standardized_package_names=package_data["standardized_package_names"])

    def link_product_package(self, product_key, package_name, props=None):
        with self.driver.session() as session:
            session.execute_write(self._link_product_package_tx, product_key, package_name, props)

    def _link_product_package_tx(self, tx, product_key, package_name, props=None):
        query = """
        MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
        MATCH (b:Package {name: $package_name})
        MERGE (p)-[r:PACKAGED_IN]->(b)
        SET r += $props
        """
        tx.run(query, product_name=product_key['pn'], manufacturer_name=product_key['manufacturer'],
               package_name=package_name, props=props)

    def link_manufacturer_package(self, manufacturer_name, package_name):
        with self.driver.session() as session:
            session.execute_write(self._link_manufacturer_package_tx, manufacturer_name, package_name)

    def _link_manufacturer_package_tx(self, tx, manufacturer_name, package_name):
        query = """
        MATCH (p:Manufacturer {name: $manufacturer_name})
        MATCH (c:Package {name: $package_name})
        MERGE (p)-[:CREATED]->(c)
        """
        tx.run(query, manufacturer_name=manufacturer_name, package_name=package_name)

    def create_measurement(self, product_key, attribute):
        with self.driver.session() as session:
            session.execute_write(self._create_measurement_tx, product_key, attribute)

    def _create_measurement_tx(self, tx, product_key, attribute):
        query = """
        MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
        MATCH (a:Attribute {name: $attribute_name})
        MERGE (p)-[:HAS_ATTRIBUTE]->(a)

        MERGE (m:Measurement {value: $value, unit: $unit, test_conditions: $test_conditions})<-[:HAS_MEASUREMENT]-(p)
        MERGE (m)-[:FOR_ATTRIBUTE]->(a)
        """
        tx.run(query,
               product_name=product_key["pn"],
               manufacturer_name=product_key["manufacturer"],
               attribute_name=attribute['name'],
               value=attribute['value'],
               unit=attribute['unit'],
               test_conditions=attribute['test_condition'])


def process_result(results, neo4j_ops):
    products = results["Products"]
    print(len(products))
    for product in products:
        product_key = {"pn": product["ManufacturerProductNumber"], "manufacturer": product["Manufacturer"]["Name"]}
        attributes = []

        neo4j_ops.create_or_update_product(product, product["Manufacturer"]["Name"], 'No')
        neo4j_ops.create_or_update_manufacturer(product_key['manufacturer'])

        datasheet_url = product['DatasheetUrl']
        if datasheet_url is not None:
            if datasheet_url.startswith('//'):
                datasheet_url = "https:" + datasheet_url
            neo4j_ops.create_or_update_datasheet(datasheet_url)
            neo4j_ops.link_product_to_datasheet(product_key, datasheet_url, "Unprocessed")
        # else:
        #     print("WAS NONE!!!!")
        neo4j_ops.link_product_manufacturer(product["ManufacturerProductNumber"],
                              product["Manufacturer"]["Name"])

        category = product["Category"]
        category_data = {"name": category["Name"], "description": category["SeoDescription"]}
        neo4j_ops.create_or_update_category(category_data)

        if len(category["ChildCategories"]) > 0:
            child_cat = category["ChildCategories"][0]
            while child_cat != "":
                child_cat_data = {"name": child_cat["Name"], "description": child_cat["SeoDescription"]}
                neo4j_ops.create_or_update_category(child_cat_data)
                neo4j_ops.link_subcategory(child_cat_data["name"], category_data["name"])
                category_data = child_cat_data
                category = child_cat
                if len(category["ChildCategories"]) > 0:
                    child_cat = category["ChildCategories"][0]
                else:
                    child_cat = ""

        neo4j_ops.link_product_category(product_key, category_data["name"])

        if product['Classifications']["MoistureSensitivityLevel"]:
            if product['Classifications']["MoistureSensitivityLevel"] != 'Not Applicable':
                attributes.append({"name": "moisture_sensitivity_level",
                                   "value": product['Classifications']["MoistureSensitivityLevel"],
                                   "description": "", "possible_symbols": "", "unit": "", "test_condition": ""})
        if product['Classifications']["RohsStatus"]:
            attributes.append({"name": "rohs_status", "value": product['Classifications']["RohsStatus"],
                               "description": "", "possible_symbols": "", "unit": "", "test_condition": ""})
        if product['Classifications']["ReachStatus"]:
            attributes.append({"name": "reach_status", "value": product['Classifications']["ReachStatus"],
                               "description": "", "possible_symbols": "", "unit": "", "test_condition": ""})
        attributes.append(
            {"name": "lead_time", "value": product['ManufacturerLeadWeeks'], "description": "",
             "possible_symbols": "", "unit": "weeks", "test_condition": ""})

        package = {"standardized_package_names": ""}
        for parameter in product["Parameters"]:
            if parameter["ParameterText"] == "Package / Case":
                package["standardized_package_names"] = parameter["ValueText"]
            elif parameter["ParameterText"] == "Supplier Device Package":
                package["package_name"] = parameter["ValueText"]
            else:
                functions = fndict[int(parameter["ParameterId"])]
                if functions is None:
                    parameter_name = snake(parameter["ParameterText"])
                    parameter_value = parameter["ValueText"]
                else:
                    if isinstance(functions[0], types.FunctionType):
                        parameter_name = functions[0](parameter["ParameterText"])
                    else:
                        parameter_name = functions[0]
                    if functions[1] is None:
                        if parameter["ValueText"] == "-":
                            parameter_value = None
                        else:
                            parameter_value = parameter["ValueText"]
                    else:
                        parameter_value = functions[1](parameter_name, parameter["ValueText"])
                if isinstance(parameter_value, str):
                    attributes.append(
                        {"name": parameter_name, "value": parameter_value,
                         "description": "", "possible_symbols": "", "unit": "", "test_condition": ""})
                elif isinstance(parameter_value, list):
                    for parameter_ in parameter_value:
                        attributes.append(
                            {"name": parameter_[0], "value": parameter_[1],
                             "description": "", "possible_symbols": "", "unit": parameter_[2],
                             "test_condition": parameter_[3]})
                else:
                    continue

        for attribute in attributes:
            if attribute['value'] is not None:
                neo4j_ops.create_or_update_attribute(attribute)
                neo4j_ops.create_measurement(product_key, attribute)
            # neo4j_ops.link_product_attribute(product["ManufacturerProductNumber"],
            #                   attribute["name"], attribute["value"])

        if 'package_name' in package.keys():
            neo4j_ops.create_or_update_package(package)
            # package_properties = {"size": " 2.9mm × 1.6mm", "junction_to_ambient_thermal_resistance": "229 °C/W"}
            package_properties = {}
            neo4j_ops.link_product_package(product_key, package["package_name"], package_properties)
            neo4j_ops.link_manufacturer_package(product["Manufacturer"]["Name"], package["package_name"])

