import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

INPUT_CSV = 'ketqua.csv'
TRANSFER_CSV = 'transfer_values.csv'
PREDICT_CSV = 'transfer_predictions.csv'
TRANSFERMARKT_URL = 'https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query='
FOOTBALLTRANSFERS_URL = 'https://www.footballtransfers.com/en/search?q='

try:
    df = pd.read_csv(INPUT_CSV, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"Error: '{INPUT_CSV}' not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

required_cols = ['Player', 'Min']
team_col = next((col for col in ['Team', 'Squad', 'team', 'TEAM'] if col in df.columns), None)
if not all(col in df.columns for col in required_cols) or not team_col:
    print(f"Error: Required columns missing. Found: {df.columns.tolist()}")
    exit()

df['Min'] = pd.to_numeric(df['Min'].astype(str).str.replace(',', ''), errors='coerce')
df = df.dropna(subset=['Min'])

players_900 = df[df['Min'] > 900][['Player', team_col, 'Min']].rename(columns={team_col: 'Team'})

def scrape_transfer_values(players_df, headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--allow-insecure-localhost")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.accept_insecure_certs = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    values = []

    for _, row in players_df.iterrows():
        player = row['Player']
        transfer_value = None

        for attempt in range(3):
            try:
                print(f"Attempting for {player}")
                driver.get(f"{TRANSFERMARKT_URL}{player.replace(' ', '+')}")
                link = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "td.hauptlink a[href*='/profil/spieler/']"))
                )
                player_url = link.get_attribute('href')
                driver.get(player_url)
                transfer_value = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/marktwertverlauf/spieler/']"))
                ).text
                print(f"Scraped {player}: {transfer_value}")
                break
            except Exception as e:
                time.sleep(5)

        if not transfer_value:
            for attempt in range(3):
                try:
                    print(f"Attempting {player}")
                    driver.get(f"{FOOTBALLTRANSFERS_URL}{player.replace(' ', '+')}")
                    transfer_value = WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.market-value"))
                    ).text
                    print(f"Scraped {player}: {transfer_value}")
                    break
                except Exception as e:
                    time.sleep(5)
            else:
                transfer_value = 'N/a'

        values.append({'player_name': player, 'transfer_value': transfer_value})
        time.sleep(3)

    driver.quit()
    return pd.DataFrame(values)

try:
    transfer_data = scrape_transfer_values(players_900, headless=True)
    transfer_data.to_csv(TRANSFER_CSV, index=False, encoding='utf-8-sig')
except Exception as e:
    print(f"Scraping failed: {e}. Using fallback data.")
    transfer_data = pd.DataFrame({'player_name': players_900['Player'], 'transfer_value': [np.nan] * len(players_900)})
    transfer_data.to_csv(TRANSFER_CSV, index=False, encoding='utf-8-sig')

merged_data = pd.merge(players_900, transfer_data, left_on='Player', right_on='player_name', how='left')

def clean_transfer_value(value):
    if value == 'N/a' or pd.isna(value):
        return np.nan
    try:
        value = value.replace('€', '').replace('£', '').strip()
        if 'm' in value.lower():
            return float(value.lower().replace('m', '')) * 1e6
        elif 'k' in value.lower():
            return float(value.lower().replace('k', '')) * 1e3
        return float(value)
    except:
        return np.nan

merged_data['transfer_value'] = merged_data['transfer_value'].apply(clean_transfer_value)

features = ['Age', 'Min', 'Gls', 'Ast', 'xG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'SoT%', 'SoT/90', 'Tkl', 'TklW', 'Blocks', 'Touches', 'Succ%', 'Fls', 'Fld', 'Won%']

def get_numeric_columns(df, cols):
    numeric_cols = []
    for col in cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col].astype(str).replace(',', ''), errors='coerce')
                if df[col].notna().sum() > 0:
                    numeric_cols.append(col)
            except:
                pass
    return numeric_cols

available_features = get_numeric_columns(df, features)
if not available_features:
    print("No valid features. Exiting.")
    exit()

df['transfer_value'] = merged_data['transfer_value']

valid_df = df.dropna(subset=available_features + ['transfer_value'])

X = valid_df[available_features]
y = valid_df['transfer_value']
players = valid_df['Player']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}, R^2: {r2:.2f}")

full_pred = model.predict(X_scaled)

predictions = pd.DataFrame({
    'Player': players.values,
    'Actual_Value': y.values,
    'Predicted_Value': full_pred
})
predictions.to_csv(PREDICT_CSV, index=False, encoding='utf-8-sig')

print(f"Done. Check '{TRANSFER_CSV}', '{PREDICT_CSV}'.")
