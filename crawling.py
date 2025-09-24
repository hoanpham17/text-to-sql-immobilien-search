#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:40:54 2025

@author: hoanpham
"""

import os

import time
import random
import json

import pandas as pd
from selenium import webdriver
import base64
import whisper # type: ignore
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager # type: ignore

from sqlalchemy.dialects.postgresql import JSONB # type: ignore
from sqlalchemy import create_engine # type: ignore
from datetime import datetime
import http.client
from dotenv import load_dotenv # type: ignore


class RealEstateScraper:
    def __init__(self, category):
        
        load_dotenv()

        self.model = whisper.load_model("large-v2")
        self.category = category
        
        # Set up database connection parameters
        self.db_host = os.getenv('DB_HOST')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_port = os.getenv('DB_PORT')
        self.connection_string = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(self.connection_string)
        
        # Set up WebDriver options
        self.chrome_options = Options()
        self.chrome_options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument("--disable-popup-blocking")

        self.service = Service(ChromeDriverManager().install())
        self.driver = None  # Initialize driver later
        
        # Define default audio file output path
        self.audio_output_path = "/Users/hoanpham/output_audio.aac"
        self.today_date = datetime.today().strftime('%Y-%m-%d')
        
        # Establish HTTP connection
        self.conn = http.client.HTTPSConnection("www.immobilienscout24.de")
        print(self.db_host, self.db_name, self.db_user, self.db_password)
        
        # Generate cookies and fetch data
        # self.generate_cookies_str()
        # self.fetch_data_from_website()
        # self.prepare_data()
    
    def generate_cookies_str(self):
        
        max_captcha_cycles = 3  
        captcha_solved_successfully = False 
        self.driver = None 
    
        
        cycle_attempt = 0
        while cycle_attempt < max_captcha_cycles and not captcha_solved_successfully:
            print(f"\n--- CAPTCHA Bypass Cycle {cycle_attempt + 1}/{max_captcha_cycles} ---")
            driver_initialized_in_cycle = False 
    
            try:
                # 1. Initialize Chrome WebDriver
                
                self.driver = webdriver.Chrome(service=self.service, options=self.chrome_options)
                driver_initialized_in_cycle = True 
    
                # 2. Navigate to the page
               
                self.driver.get(f"https://www.immobilienscout24.de/Suche/de/{self.category}?pagenumber=3")
                time.sleep(4) # 
    
                
                max_inner_attempts = 3
                captcha_solved_in_cycle = False
    
                
                #print(f"Starting CAPTCHA solving attempts ({max_inner_attempts} within this cycle)...")
                for attempt_num_inner in range(max_inner_attempts):
                    print(f"    Attempt {attempt_num_inner + 1}/{max_inner_attempts}...")
                    captcha_solved_in_cycle = self.attempt_solve_aws_captcha()
                    if captcha_solved_in_cycle:
                        print("    CAPTCHA solved successfully!")
                        break 
                    else:
                        print("    CAPTCHA solving failed.")
                        if attempt_num_inner < max_inner_attempts - 1:
                             time.sleep(5) 
    
                if captcha_solved_in_cycle:
                    print("CAPTCHA bypassed successfully in this cycle. Getting data...")
                    captcha_solved_successfully = True 
    
                    # get number of pages and cookie
                    try:
                        time.sleep(3) 
                        self.number_of_pages = self.driver.execute_script(
                            '''return window.IS24.resultList.resultListModel.searchResponseModel["resultlist.resultlist"].paging.numberOfPages;'''
                        )
                        print(f"Number of pages found: {self.number_of_pages}")
    
                        cookies = self.driver.get_cookies()
                        self.cookies_str = "; ".join([f"{cookie['name']}={cookie['value']}" for cookie in cookies])
                        print("Cookies generated successfully.")
    
                    except Exception as post_solve_e:
                        print(f"Error occurred AFTER CAPTCHA was solved (getting pages/cookies): {post_solve_e}")
                        
                else: 
                    print(f"CAPTCHA failed after {max_inner_attempts} attempts in cycle {cycle_attempt + 1}. Preparing for next cycle.")                 
    
            except Exception as cycle_e:                
                print(f"An unexpected error occurred during CAPTCHA cycle {cycle_attempt + 1}: {cycle_e}")
    
            finally:
               
                if driver_initialized_in_cycle and self.driver:
                    
                    self.driver.quit()
                    self.driver = None 
    
            cycle_attempt += 1 
    
        print("\n--- Final Status ---")
        if captcha_solved_successfully:
            print("CAPTCHA bypass process finished.")
        else:
            print(f"CAPTCHA bypass failed after {max_captcha_cycles} total cycles.")
            self.cookies_str = None 
            self.number_of_pages = 0 
   

    def generate_random_user_agent(self):
        """Generate a random user-agent string."""
        user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:106.0) Gecko/20100101 Firefox/106.0",
            "Mozilla/5.0 (iPhone; CPU OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Vivaldi/5.6.2753.48",
            "Mozilla/5.0 (iPad; CPU OS 16_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (BlackBerry; U; BlackBerry 9900; en) AppleWebKit/534.8 (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.8",
            "Mozilla/5.0 (Linux; Android 12; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (Windows NT 5.1; rv:78.0) Gecko/20100101 Firefox/78.0",
        ]
        return random.choice(user_agents)
    
    def fetch_data_from_website(self):
        """Fetch data from the ImmobilienScout24 website."""
        data_list = []
        page = 1
        retries = 3

        # while page <= 100:
        while page <= self.number_of_pages:
            success = False
            while not success and retries > 0:
                try:
                    user_agent = self.generate_random_user_agent()
                    headers = {
                        'cookie': self.cookies_str,
                        'accept': "application/json",
                        'accept-language': "en-US,en;q=0.9,vi;q=0.8",
                        'user-agent': user_agent
                    }

                    url = f"/Suche/de/{self.category}?pagenumber={page}"
                    self.conn.request("GET", url, "", headers)
                    res = self.conn.getresponse()
                    data = res.read().decode("utf-8")
                    json_data = json.loads(data)
                    data_list.append(json_data)
                        
                    print(f"Page {page} data collected.")
                    success = True
                    

                    #time.sleep(random.uniform(0.25, 0.71))
                    page += 1

                except json.JSONDecodeError:
                    print(f"Failed to decode JSON on page {page}, retrying...")
                    retries -= 1
                    time.sleep(1)
                except Exception as e:
                    print(f"Error on page {page}: {e}, retrying...")
                    retries -= 1
                    time.sleep(1)
                if not success:
                    self.generate_cookies_str()

        self.data_list = data_list
        
    def click_audio_captcha_button(self):
        """Click on the audio CAPTCHA button."""
        try:
            captcha_container = WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "captcha-container"))
            )
            print("CAPTCHA container found.")

            shadow_root = self.driver.execute_script(
                "return document.querySelector('#captcha-container > awswaf-captcha').shadowRoot"
            )
            time.sleep(2)
            audio_button = shadow_root.find_element(
                By.CSS_SELECTOR, "#amzn-btn-audio-internal")
            time.sleep(2)
            audio_button.click()
            time.sleep(2)
            print("Audio button clicked!")

        except Exception as e:
            print(f"Error: {e}")

    def get_audio_and_transcribe(self):
        """Retrieve and transcribe audio CAPTCHA."""
        try:
            audio_element = self.driver.execute_script('''
                return document.querySelector("#captcha-container > awswaf-captcha")
                .shadowRoot.querySelector("#root > div > form > div:nth-child(2) > div > audio");
            ''')
            audio_source = audio_element.get_attribute('src')

            if 'base64,' in audio_source:
                base64_audio = audio_source.split('base64,')[1]
                with open(self.audio_output_path, 'wb') as audio_file:
                    audio_file.write(base64.b64decode(base64_audio))
                print(f"Audio file saved to {self.audio_output_path}")

                result = self.model.transcribe(self.audio_output_path)
                transcription = result["text"].strip()
                return transcription
            else:
                print("Audio source is not in base64 format or might be a URL.")
                return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    def extract_first_word(self, text):
        """Extract the first word after the first period in the transcription."""
        parts = text.split('.')
        if len(parts) > 1:
            first_word = parts[1].strip().split()[0]
            return first_word.lower().strip('., ')
        return "No word found after the first period."

    def solve_audio_captcha(self, first_word):
        self.driver.execute_script(f'''
            var input_field = document.querySelector("#captcha-container > awswaf-captcha")
            .shadowRoot.querySelector("#root > div > form > div:nth-child(2) > div > div:nth-child(6) > div:nth-child(2) > input[type=text]");
            input_field.value = "{first_word}";
        ''')
        self.driver.execute_script('''
            var verify_button = document.querySelector("#captcha-container > awswaf-captcha")
            .shadowRoot.querySelector("#amzn-btn-verify-internal");
            verify_button.click();
        ''')

    def attempt_solve_aws_captcha(self):
        try:
            self.click_audio_captcha_button()

            print("Waiting for audio source to be ready...")
            try:
                # Wait for the audio element to exist 
                # Accessing shadow DOM within WebDriverWait using execute_script
                WebDriverWait(self.driver, 15).until(
                     lambda driver: driver.execute_script('''
                         var audio = document.querySelector("#captcha-container > awswaf-captcha")
                         ?.shadowRoot?.querySelector("#root > div > form > div:nth-child(2) > div > audio");
                         return audio && audio.getAttribute('src') && audio.getAttribute('src').includes('base64,');
                     ''')
                )
                print("Audio source is ready (base64 found).")
            except Exception as e:
                print(f"Error waiting for audio source: {e}")
                print("Audio source not available within timeout. CAPTCHA may not have loaded correctly.")
                return False 

            transcription = self.get_audio_and_transcribe()

            if transcription and transcription != "No word found after the first period.": 
                 first_word = self.extract_first_word(transcription)
                 print(f"Transcription: '{transcription}'")
                 print(f"Extracted first word: '{first_word}'")

                 if first_word and first_word != "No word found after the first period.":
                 
                    self.solve_audio_captcha(first_word)
                    print("Submitted CAPTCHA solution.")

                    print("Waiting for CAPTCHA to disappear...")
                    try:
                        WebDriverWait(self.driver, 20).until(
                             EC.invisibility_of_element_located((By.ID, "captcha-container"))
                        )
                        print("CAPTCHA disappeared - likely solved.")
                        return True 
                    except Exception as e:
                        print(f"CAPTCHA did not disappear within timeout: {e}")
                        print("CAPTCHA solution may have failed or page is loading slowly.")
                        return False

                 else:
                    print("Could not extract valid word from transcription.")
                    return False 
            else:
                print("Transcription failed or returned no valid text.")
                return False 

        except Exception as e:
            print(f"An error occurred during CAPTCHA solving process: {e}")
            return False 


    def prepare_data(self):
        """Convert the fetched JSON data into a DataFrame."""
        all_entries = []
        for data in self.data_list:
            try:
                entries = data["searchResponseModel"]["resultlist.resultlist"]["resultlistEntries"][0]["resultlistEntry"]
                for entry in entries:
                    all_entries.append(entry)
            except KeyError as e:
                print(f"Key error: {e}")
            except IndexError as e:
                print(f"Index error: {e}")

        df = pd.DataFrame(all_entries)
        df['update_date'] = datetime.today().strftime('%Y-%m-%d')
        self.df = df  # Store DataFrame in class
        return df


    def upload_to_raw(self, table_name: str):
        try:
            # Detect dict/list columns and leave them as-is (SQLAlchemy will handle them as JSONB)
            jsonb_columns = []
            for col in self.df.columns:
                if self.df[col].apply(lambda x: isinstance(x, (dict, list))).any():
                    jsonb_columns.append(col)
    
            # Define SQLAlchemy dtype mapping: JSONB for dict/list columns
            dtype = {col: JSONB for col in jsonb_columns}
    
            # Upload to PostgreSQL
            self.df.to_sql(
                name=table_name,
                con=self.engine,
                schema='raw',
                if_exists='replace',
                index=False,
                dtype=dtype
            )
    
            print(f"✅ Data uploaded to raw.{table_name} with correct JSONB types.")
    
        except Exception as e:
            print(f"❌ Upload failed: {e}")
           



  
#---------------------------


# Crawling for haus-kaufen
 

scraper = RealEstateScraper("haus-kaufen")
scraper.generate_cookies_str()
scraper.fetch_data_from_website()
scraper.number_of_pages
df = scraper.prepare_data()
scraper.upload_to_raw('haus_kaufen')

