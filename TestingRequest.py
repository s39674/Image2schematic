
from asyncore import read
import requests
import json
from PyPDF2 import PdfReader
import io

ICname = "TC1427"

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.5',
    #'Referer': 'https://octopart.com/search?q=TC1427&currency=USD&specs=0',
    # Already added when you pass json=
    # 'content-type': 'application/json',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    # Requests doesn't support trailers
    # 'TE': 'trailers',
}

json_data = {
    'operationName': 'PricesViewSearch',
    'variables': {
        'currency': 'USD',
        'filters': {},
        'in_stock_only': False,
        'limit': 10,
        'q': ICname,
        'start': 0,
    },
    'query': 'query PricesViewSearch($currency: String!, $filters: Map, $in_stock_only: Boolean, $limit: Int!, $q: String, $sort: String, $sort_dir: SortDirection, $start: Int) {\n  search(currency: $currency, filters: $filters, in_stock_only: $in_stock_only, limit: $limit, q: $q, sort: $sort, sort_dir: $sort_dir, start: $start) {\n    all_filters {\n      group\n      id\n      name\n      shortname\n      __typename\n    }\n    applied_category {\n      ancestors {\n        id\n        name\n        path\n        __typename\n      }\n      id\n      name\n      path\n      __typename\n    }\n    applied_filters {\n      display_values\n      name\n      shortname\n      values\n      __typename\n    }\n    results {\n      _cache_id\n      aka_mpn\n      description\n      part {\n        _cache_id\n        best_datasheet {\n          url\n          __typename\n        }\n        best_image {\n          url\n          __typename\n        }\n        cad {\n          add_to_library_url\n          footprint_image_url\n          has_3d_model\n          has_altium\n          has_eagle\n          has_kicad\n          has_orcad\n          symbol_image_url\n          __typename\n        }\n        cad_models {\n          has_3d_model\n          has_symbol\n          has_footprint\n          __typename\n        }\n        cad_request_url\n        category {\n          id\n          __typename\n        }\n        counts\n        descriptions {\n          text\n          __typename\n        }\n        free_sample_url\n        id\n        manufacturer {\n          id\n          is_verified\n          name\n          __typename\n        }\n        manufacturer_url\n        median_price_1000 {\n          _cache_id\n          converted_currency\n          converted_price\n          __typename\n        }\n        mpn\n        sellers {\n          _cache_id\n          company {\n            homepage_url\n            id\n            is_distributorapi\n            is_verified\n            name\n            slug\n            __typename\n          }\n          is_broker\n          is_rfq\n          offers {\n            _cache_id\n            click_url\n            id\n            inventory_level\n            moq\n            packaging\n            prices {\n              _cache_id\n              conversion_rate\n              converted_currency\n              converted_price\n              currency\n              price\n              quantity\n              __typename\n            }\n            sku\n            updated\n            __typename\n          }\n          __typename\n        }\n        series {\n          id\n          name\n          url\n          __typename\n        }\n        slug\n        v3uid\n        __typename\n      }\n      __typename\n    }\n    suggested_categories {\n      category {\n        id\n        name\n        path\n        __typename\n      }\n      count\n      __typename\n    }\n    suggested_filters {\n      group\n      id\n      name\n      shortname\n}}\n}\n',
}

response = requests.post('https://octopart.com/api/v4/internal', headers=headers, json=json_data)

JSON_response = json.loads(response.text)

#print(JSON_response)



for result in JSON_response['data']['search']['results']:
    
    part = result['part']
    Descriptions = part['descriptions']

    part_description = part['descriptions'][0]['text']
    for description in Descriptions:
        if description['text'].find(part['mpn']) > 0:
            part_description = description['text']
            break

    print(f"part name: {part['mpn']}\nPart description: {part_description}\n best datasheet: {part['best_datasheet']['url']}\n")


FirstBestDataSheetUrl = JSON_response['data']['search']['results'][0]['part']['best_datasheet']['url']

User_input1 = input(f"Should I try to extract info from: {FirstBestDataSheetUrl} ? [y | n] ")
if User_input1 == "y":
    response = requests.get(FirstBestDataSheetUrl)
    reader2 = PdfReader(io.BytesIO(response.content))
    if len(reader2.pages) > 0:
        for page in reader2.pages:
            if "PIN FUNCTION TABLE" in page.extract_text():
                print(page.extract_text())