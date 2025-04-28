import sys
from neo4j import GraphDatabase
from nexarClient import NexarClient
from pymongo import MongoClient
from datetime import datetime

# Set up start
MONGO_URI = "mongodb://admin:password1234@localhost:27017/"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password1234"

NEXAR_CLIENT_ID = ""
NEXAR_CLIENT_SECRET = ""
# Set up end

client = MongoClient(MONGO_URI)
db = client["MPN"]
parts_collection = db["parts"]
snapshots_collection = db["snapshots"]

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


QUERY_MPN = '''
    query specAttributes($mpn: String!){
      supSearchMpn(
        q: $mpn,
        limit: 1
        ) {
        hits
        results {
          part {
            mpn
            shortDescription
            manufacturer {
              name
            }
            documentCollections {
              name
              documents {
                name
                pageCount
                url
                }
            }
            bestDatasheet {
                name
                pageCount
                url
                }
            referenceDesigns {
            name
            url
            }
            specs {
              attribute {
                shortname
                group
              }
              displayValue
            }      
          }
        }
      }
    }
'''

QUERY_PRICE = '''
    query pricingByVolumeLevels($mpn: String!, $manufacturer_filter: Map!){
      supSearch(
        q: $mpn, 
        filters: $manufacturer_filter,
        limit: 1) {
        hits
        results {
          part {
            name
            mpn
            manufacturer {
              name
            }
            sellers {
              company {
                name
              }
              offers {
                packaging
                inventoryLevel
                moq
                factoryLeadDays
                factoryPackQuantity
                prices {
                  quantity
                  price
                  currency
                }
              }
            }      
          }
        }
      }
    }
'''

def save_pricing_data(results, driver):
    """Save pricing data to MongoDB parts and snapshots collections"""
    if not results or 'supSearch' not in results or 'results' not in results['supSearch']:
        print("No valid data to save")
        return

    if results['supSearch']['results'] is None:
        print("No valid data to save")
        return

    current_time = datetime.now()

    snapshot_date = datetime(current_time.year, current_time.month, 1)

    for item in results['supSearch']['results']:
        part_data = item.get('part', {})
        if not part_data or 'mpn' not in part_data:
            continue

        mpn = part_data.get('mpn')
        name = part_data.get('name', '')
        manufacturer = part_data.get('manufacturer', {}).get('name', '')

        sellers_data = []
        for seller in part_data.get('sellers', []):
            seller_name = seller.get('company', {}).get('name', '')
            offers = []

            for offer in seller.get('offers', []):
                offers.append({
                    'packaging': offer.get('packaging'),
                    'inventoryLevel': offer.get('inventoryLevel'),
                    'moq': offer.get('moq'),
                    'factoryLeadDays': offer.get('factoryLeadDays'),
                    'factoryPackQuantity': offer.get('factoryPackQuantity'),
                    'prices': offer.get('prices', [])
                })

            if seller_name and offers:
                sellers_data.append({
                    'name': seller_name,
                    'offers': offers
                })

        part_document = {
            'mpn': mpn,
            'name': name,
            'manufacturer': {'name': manufacturer},
            'sellers': sellers_data,
            'lastUpdated': current_time
        }

        snapshot_document = {
            'mpn': mpn,
            'snapshotDate': snapshot_date,
            'snapshotType': 'monthly',
            'manufacturer': {'name': manufacturer},
            'sellers': sellers_data
        }

        parts_collection.update_one(
            {'mpn': mpn},
            {'$set': part_document},
            upsert=True
        )

        existing_snapshot = snapshots_collection.find_one({
            'mpn': mpn,
            'snapshotDate': snapshot_date
        })

        if not existing_snapshot:
            snapshots_collection.insert_one(snapshot_document)
            print(f"Created new monthly snapshot for {mpn}")
        else:
            print(f"Monthly snapshot already exists for {mpn}")
        update_price_status(driver, name, manufacturer)
        print(f"Updated data for {mpn}")


def update_price_status(driver, product_name, manufacturer_name):
    query = """
    MATCH (p:Product {name: $product_name, manufacturer: $manufacturer_name})
    SET p.price_processed = "Yes"
    """
    with driver.session() as session:
        result = session.execute_write(lambda tx: tx.run(query, product_name=product_name, manufacturer_name=manufacturer_name).data())
    print(f"Update {product_name} status.")

def get_active_products_by_category(driver, category_name):
    query = """
    MATCH (p:Product)-[:MANUFACTURED_BY]->(m:Manufacturer)
    MATCH (p)-[:BELONGS_TO]->(c:Category {name: $category_name})
    WHERE p.status = "Active"
    RETURN p.name AS product, m.name AS manufacturer
    """

    products = []
    with driver.session() as session:
        result = session.execute_read(lambda tx: tx.run(query, category_name=category_name).data())
        for record in result:
            products.append((record["product"], record["manufacturer"]))

    return products


manuf_dict = {"Analog Devices Inc.": "Analog Devices", "Advanced Micro Devices": "AMD", "Renesas Electronics Corporation": "Renesas"}


if __name__ == '__main__':

    clientId = NEXAR_CLIENT_ID
    clientSecret = NEXAR_CLIENT_SECRET
    nexar = NexarClient(clientId, clientSecret)

    mode = int(input('Please select the mode (1-MPN parse, 2 - from graph category): '))

    while (True):
        if mode == 1:
            mpn = input('Search MPN: ')
            manufacturer_id = input('Manufacturer ID: ')

            manufacturer_filter = {"manufacturer_id": [manufacturer_id]}

            if not mpn:
                sys.exit()
            variables = {
                'mpn': mpn,
                'manufacturer_filter': manufacturer_filter
            }
            print(variables)
            results = nexar.execute_query(QUERY_PRICE, variables)

            if results:
                print(results)
                save_pricing_data(results, driver)
            else:
                print('Sorry, no parts found')
                print()
        if mode == 2:
            category = input('Search category: ')

            products = get_active_products_by_category(driver, category)

            print(len(products))

            for product in products:
                product_name, manufacturer_id = product
                if manufacturer_id in manuf_dict.keys():
                    manufacturer_id = manuf_dict[manufacturer_id]
                manufacturer_filter = {"manufacturer_id": [manufacturer_id]}

                variables = {
                    'mpn': product_name,
                    'manufacturer_filter': manufacturer_filter
                }
                results = nexar.execute_query(QUERY_PRICE, variables)
                save_pricing_data(results, driver)
                print(product_name, manufacturer_filter, results)
