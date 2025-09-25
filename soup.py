""" import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

url = 'https://www.scrapethissite.com/pages/simple/'

def get_soup(url):
    # Takes URL and returns a soup object
    resp = requests.get(url)
    if resp.status_code == 200:
        return bs(resp.text, 'lxml')
    return None

def get_details(country_div):
    # Extract details from one country div
    country_name = country_div.find('h3', class_='country-name').text.strip()
    capital_name = country_div.find('span', class_='country-capital').text.strip()
    population = country_div.find('span', class_='country-population').text.strip()
    area = country_div.find('span', class_='country-area').text.strip()
    return [country_name, capital_name, population, area]

def get_countries(url):
    # Extract details from all the countries
    soup = get_soup(url)
    country_all = soup.find_all('div', class_="col-md-4 country")
    countries = [get_details(country) for country in country_all]
    return countries

# Get data
countries = get_countries(url)

# Convert to DataFrame
df = pd.DataFrame(countries, columns=['Country', 'Capital', 'Population', 'Area'])
print(df)
 """


import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import time

BASE_URL = "https://www.scrapethissite.com/pages/simple/"

def get_soup(url):
    resp = requests.get(url)
    if resp.status_code == 200:
        return bs(resp.text, 'lxml')
    return None

def get_details(country_div):
    """Extract details from one country div"""
    country_name = country_div.find('h3', class_='country-name').text.strip()
    capital_name = country_div.find('span', class_='country-capital').text.strip()
    population = country_div.find('span', class_='country-population').text.strip()
    area = country_div.find('span', class_='country-area').text.strip()
    return [country_name, capital_name, population, area]

def scrape_all_pages(base_url):
    all_countries = []
    url = base_url
    
    while url:
        soup = get_soup(url)
        country_divs = soup.find_all('div', class_='col-md-4 country')
        all_countries.extend([get_details(div) for div in country_divs])

        time.sleep(1)
        
        # Check for next page link
        next_link = soup.find('a', class_='next')
        if next_link and 'href' in next_link.attrs:
            url = base_url.rstrip('/') + next_link['href']
        else:
            url = None  # stop when no more pages
    
    return all_countries

# Run scraper
countries = scrape_all_pages(BASE_URL)

# Convert to DataFrame
df = pd.DataFrame(countries, columns=['Country', 'Capital', 'Population', 'Area'])
print(df)

# Save to CSV
df.to_csv("countries.csv", index=False)
