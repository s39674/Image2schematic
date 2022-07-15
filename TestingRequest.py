
import re
import sys
from bs4 import BeautifulSoup
import requests
import json
#from PyPDF2 import PdfReader


webSiteUrl = "https://www.findchips.com"
#webSiteUrl = "https://octopart.com"

ICname = "TC1427"

def fetchComponentInfo(ICname: str, webSiteUrl: str = "https://octopart.com", userAgent: str = 'Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0', cookies = {}):

    # Golbal headers for any site
    headers = {
        'User-Agent': userAgent,
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'Accept-Language': 'en-US,en;q=0.5',
        'Sec-Fetch-Dest': 'empty',
        'referer': webSiteUrl,
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
    }

    if webSiteUrl.endswith("octopart.com"):
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

        response = requests.post(f'{webSiteUrl}/api/v4/internal', headers=headers, json=json_data, cookies=cookies)
        if response.status_code != 200:
            print(f"Got an invalid status code from {webSiteUrl} ! status code: {response.status_code}")
            return

        JSON_response = json.loads(response.text)

        """
        # for quering every part:
        for result in JSON_response['data']['search']['results']:
            
            part = result['part']
            Descriptions = part['descriptions']
            slug = part['slug']

            # getting valid description
            part_description = part['descriptions'][0]['text']
            for description in Descriptions:
                if description['text'].find(part['mpn']) > 0:
                    part_description = description['text']
                    break

            print(f"\n-------------\npart name: {part['mpn']}\nPart description: {part_description}\n best datasheet: {part['best_datasheet']['url']}\nOctopart page: {webSiteUrl}/{slug}")
        """

        bestResult = JSON_response['data']['search']['results'][0]
        part = bestResult['part']
        Descriptions = part['descriptions']
        slug = part['slug']

        # getting valid description
        part_description = part['descriptions'][0]['text']
        for description in Descriptions:
            if description['text'].find(part['mpn']) > 0:
                part_description = description['text']
                break

        response = requests.get(f"{webSiteUrl}{slug}",
            params={
                'r': 'sp',
            }, 
            headers=headers
        )


        numOfPins = 0
        if response.status_code != 200:
            print(f"Note: got invalid status code: {response.status_code} for part page")
        else:
            soup = BeautifulSoup(response.text,"lxml")
            
            for items in soup.select("tr"):
                data = [item.get_text(strip=True) for item in items.select("th,td")]
                if data[0] == "Number of Pins" or data[0] == "Number of Terminals":
                    numOfPins = data[1]
                    break

            """
            # pdf extraction, may be used in the future.
            FirstBestDataSheetUrl = JSON_response['data']['search']['results'][0]['part']['best_datasheet']['url']

            User_input1 = input(f"Should I try to extract info from: {FirstBestDataSheetUrl} ? [y | n] ")
            if User_input1 == "y":
                response = requests.get(FirstBestDataSheetUrl)
                reader2 = PdfReader(io.BytesIO(response.content))
                if len(reader2.pages) > 0:
                    for page in reader2.pages:
                        if "PIN FUNCTION TABLE" in page.extract_text():
                            print(page.extract_text())
            """
        
        return part['mpn'], part_description, part['best_datasheet']['url'], webSiteUrl + slug, numOfPins

    elif webSiteUrl.endswith("findchips.com"):
        # Getting details page for a part
        response = requests.get(f'{webSiteUrl}/search/{ICname}', headers=headers)
        if response.status_code != 200:
            print(f"Got an invalid status code from {webSiteUrl} ! status code: {response.status_code}")
            return
        soup = BeautifulSoup(response.text,"lxml")

        firstMatch = soup.find_all('div', class_=re.compile("full-details"))[0]
        partPage = webSiteUrl+firstMatch.a['href']

        # Getting details for part
        response = requests.get(partPage, headers=headers)
        if response.status_code != 200:
            #print(f"Got an invalid status code from {webSiteUrl}{partUrl} ! status code: {response.status_code}")
            print(f"Got an invalid status code from {partPage} ! status code: {response.status_code}") 
            return

        soup = BeautifulSoup(open("temp.html",'r'),"lxml")

        # TODO: find a way to get datasheet url from that site
        partName = None 
        numOfPins = None
        
        partDetails = soup.find_all('small')
        for partDetail in partDetails:
            # Could get more info here; use print(partDetail.text) to see diffrent stuff to grab
            if partDetail.text.strip() == "Number of Terminals:":
                numOfPins = partDetail.parent.p.text.strip() # NOTE: str
            
            elif partDetail.text.strip() == "Source Content uid:":
                partName = partDetail.parent.p.text.strip()
        
        return partName, None, None, partPage, numOfPins
    
    
    print(f"Can't find part Details. website url: {webSiteUrl}")
    return


name, partDescription, bestDatasheetUrl, partPage, numOfPins = fetchComponentInfo(ICname, webSiteUrl)

print(f"\n-------------\nBest part found:\n \
    name: {name}\n \
    Part description: {partDescription}\n \
    best datasheet: {bestDatasheetUrl}\n \
    part page: {partPage}\n \
    Number of pins: {numOfPins}")