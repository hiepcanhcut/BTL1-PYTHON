from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import pandas as pd

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

url = 'https://fbref.com/en/comps/9/stats/Premier-League-Stats'

try:
    driver.get(url)
    time.sleep(5)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    table = soup.find('table', {'id': 'stats_standard'})

    players_over_90 = []

    for row in table.find('tbody').find_all('tr'):
        name_col = row.find('th', {'data-stat': 'player'}) or row.find('td', {'data-stat': 'player'})
        minutes_col = row.find('td', {'data-stat': 'minutes'})
        
        if name_col and minutes_col:
            try:
                minutes = int(minutes_col.text.replace(',', ''))
                if minutes > 90: 
                    player_info = {
                        'player': name_col.text.strip(),
                    }
                    for stat in row.find_all('td'):
                        stat_name = stat.get('data-stat')
                        if stat_name:
                            player_info[stat_name] = stat.text.strip()

                    players_over_90.append(player_info)

            except ValueError:
                continue

    df = pd.DataFrame(players_over_90)

    df = df.rename(columns={
                'player': 'Player',
                'nationality' : 'Nation',
                'position' : 'Pos',
                'team': 'Squad',
                'Age': 'Age',
                'Born': 'Born',
                'age': 'Age',
                'games': 'MP',
                'games_starts': 'Starts',
                'minutes' : 'Min',
                'minutes_90s': '90s',
                'goals': 'Gls',
                'assists': 'Ast',
                'goals_assists': 'G+A',
                'goals_pens': 'G-PK',
                'pens_made': 'PK',
                'pens_att': 'PKatt',
                'cards_yellow': 'CrdY',
                'cards_red': 'CrdR',
                'xg': 'xG',
                'npxg': 'npxG',
                'xg_assist': 'xAG',
                'npxg_xg_assist': 'npxG+xAG',
                'progressive_carries': 'PrgC',
                'progressive_passes': 'PrgP',
                'progressive_passes_received': 'PrgR',
                'goals_per90': 'Gls',
                'assists_per90': 'Ast',
                'goals_assists_per90': 'G+A',
                'goals_pens_per90': 'G-PK',
                'goals_assists_pens_per90': 'G+A-PK',
                'xg_per90': 'xG',
                'xg_assist_per90': 'xAG',
                'xg_xg_assist_per90': 'xG+xAG',
                'npxg_per90': 'npxG',
                'npxg_xg_assist_per90': 'npxG+xAG',
                'matches': 'Matches'
            })

    df = df.sort_values(by='Player', ascending=True)

    print(df)

    df.to_csv('ketqua.csv', index=False, encoding='utf-8-sig')

finally:
    driver.quit()
