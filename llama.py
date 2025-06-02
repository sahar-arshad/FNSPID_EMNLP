import os
import pandas as pd
import ollama
import time
from tqdm import tqdm
import re
from math import ceil
from datetime import timedelta

#CSV_URL = "https://raw.githubusercontent.com/sahar-arshad/psx_news/main/cement/Cement_2021_1(1595).csv"
CSV_URL = "psx_train/oil/Oil_2021_1(1052).csv"

print(f" Loading CSV from {CSV_URL}")
df = pd.read_csv(CSV_URL)
print(f" Loaded {len(df)} rows and {len(df.columns)} columns.")
print("\n🔍 Preview of the first 5 rows:")
df['sentiment_score'] = None

print(df.head(2))
#print("Stopping here.")
#exit()  # Nothing after this will run

FEWSHOT_PROMPT = """
Forget all previous instructions.

You are a financial expert with extensive experience in stock recommendation and market based news sentiment analysis. You will receive summarized news for a specific stock and its stock symbol.
Your task is to analyze the overall news in the context of the stock’s potential short-term movement, and return a sentiment score from 1 to 5, based on the expected directional impact on the company’s stock price. The scoring criteria lies in one of the following score bands:

Scoring Criteria:
Score 5 – Strongly Positive: The news is likely to significantly increase investor confidence and drive the stock price up.
Score 4 – Moderatly Positive: The news is moderately positive, potentially causing a small upward price movement.Score 3 – Neutral: The news is balanced or has no clear market impact.
Score 2 – Moderately Negative: The news may cause a small decline in stock price.
Score 1 – Strongly Negative: The news is likely to significantly decrease investor confidence and drive the stock price down.

Below are a few examples for reference:

Examples:
Stock: APPL
Text:"Below is Validea's guru fundamental report for APPLE INC (AAPL). Of the 22 guru strategies we follow, AAPL rates highest using our Twin Momentum Investor model based on the published strategy of Dashan Huang. This momentum model looks for a combination of fundamental momentum and price momentum.
APPLE INC (AAPL) is a large-cap growth stock in the Communications Equipment industry. The rating using this strategy is 100% based on the firmâ€™s underlying fundamentals and the stockâ€™s valuation. A score of 80% or above typically indicates that the strategy has some interest in the stock and a score above 90% typically indicates strong interest.
The following table summarizes whether the stock meets each of this strategy's tests. Not all criteria in the below table receive equal weighting or are independent, but the table provides a brief overview of the strong and weak points of the security in the context of the strategy's criteria.
FUNDAMENTAL MOMENTUM: PASS
TWELVE MINUS ONE MOMENTUM: PASS
FINAL RANK: PASS
Detailed Analysis of APPLE INC
AAPL Guru Analysis"
Sentiment Score:4

Stock:EBAY
Text:"Fool.com contributor Parkev Tatevosian reveals his top dividend stocks to buy in December.
*Stock prices used were the afternoon prices of Nov. 30, 2023. The video was published on Dec. 3, 2023.
10 stocks we like better than AT&T. When our analyst team has a stock tip, it can pay to listen. After all, the newsletter they have run for over a decade, Motley Fool Stock Advisor, has tripled the market.*
They just revealed what they believe are the ten best stocks for investors to buy right now... and AT&T wasn't one of them! That's right -- they think these 10 stocks are even better buys.
See the 10 stocks
*Stock Advisor returns as of November 29, 2023
Parkev Tatevosian, CFA has positions in 3M. The Motley Fool has positions in and recommends Microsoft and Target. The Motley Fool recommends 3M, Deere, and eBay and recommends the following options: short January 2024 $45 calls on eBay. The Motley Fool has a disclosure policy.
Parkev Tatevosian is an affiliate of The Motley Fool and may be compensated for promoting its services. If you choose to subscribe through his link, he will earn some extra money that supports his channel. His opinions remain his own and are unaffected by The Motley Fool.
The views and opinions expressed herein are the views and opinions of the author and do not necessarily reflect those of Nasdaq, Inc."
Sentiment Score:3

Stock:AAPL
Text:"In a letter to the Department of Justice, Senator Ron Wyden said foreign officials were demanding the data from Alphabet's GOOGL.O Google and Apple AAPL.O. By Raphael Satter WASHINGTON, Dec 6 (Reuters) - Unidentified governments are surveilling smartphone users via their apps' push notifications, a U.S. senator warned on Wednesday. In a statement, Apple said that Wyden's letter gave them the opening they needed to share more details with the public about how governments monitored push notifications."
Sentiment Score:1

Now analyze and Please respond strictly in this format: Sentiment Score: <number>:
Stock: {ticker}
Text: {text}
Sentiment:
"""

import re

def extract_score_from_response(response_text):
    text = response_text.strip().lower()

    # 1. Look for "Sentiment Score: <number>"
    match = re.search(r'sentiment\s*score(?:\s*of)?\s*[:\-]?\s*(\d)', text)
    if match:
        return int(match.group(1))

    # 2. Look for just "Score: <number>"
    match = re.search(r'\bscore\s*[:\-]?\s*(\d)', text)
    if match:
        return int(match.group(1))

    # 3. Look for "assign a sentiment score of <number>"
    match = re.search(r'assign(?:ing)?\s+(?:a\s+)?sentiment\s+score\s+(?:of\s+)?(\d)', text)
    if match:
        return int(match.group(1))

    # 4. Fallback: last standalone digit 1–5 in entire response
    matches = re.findall(r'\b([1-5])\b', text)
    if matches:
        return int(matches[-1])  # return last one

    # 5. Default to 3 if all fails
    return 3

# Function to query Ollama
def get_sentiment_score(text, ticker):
    prompt = FEWSHOT_PROMPT.format(text=text, ticker=ticker)
    try:
        response = ollama.chat(model='llama3.1:8b', messages=[{"role": "user", "content": prompt}])
        result = response['message']['content']
        print(result)
        return extract_score_from_response(result)
    except Exception as e:
        print(f"Error: {e}")
        return None

# Setup
output_folder = 'psx_train/sentiment_psx/Oil_2021_1(1052)'
os.makedirs(output_folder, exist_ok=True)


# Batch logic
BATCH_SIZE = 25
total_records = len(df)
#total_batches = min(5, (total_records + BATCH_SIZE - 1) // BATCH_SIZE)  # Limit to 3 batches
total_batches = ceil(total_records / BATCH_SIZE)
print(f" Total records: {total_records} | Batch size: {BATCH_SIZE} | Total batches: {total_batches}")


start_time = time.time()

for batch_num in range(total_batches):
    batch_start_time = time.time()
    start = batch_num * BATCH_SIZE
    end = min((batch_num + 1) * BATCH_SIZE, total_records)

    print(f"\n Processing batch {batch_num + 1} of {total_batches} (rows {start} to {end - 1})")

    batch_df = df.iloc[start:end].copy()
    batch_filename = f"batch_{batch_num+1:02}.csv"
    batch_path = os.path.join(output_folder, batch_filename)

    # Skip if already exists
    if os.path.exists(batch_path):
        print(f" Batch already exists at {batch_path}. Skipping.")
        continue

    for i in batch_df.index:
        text = batch_df.at[i, 'TextRank_summary']
        ticker = batch_df.at[i, 'Stock_symbol']
        score = get_sentiment_score(text, ticker)
        batch_df.at[i, 'sentiment_score'] = score
        time.sleep(1.0)  # Throttling

    # Save batch
    print(batch_df.head(5))
    batch_df.to_csv(batch_path, index=False)
    print(f" Saved batch to {batch_path}")

    # Git push logic
#    os.system(f'git add "{batch_path}"')
#    os.system(f'git commit -m " Added {batch_filename} for tech_2021(1392)"')
#    os.system("git push origin master")  # or 'main' if your branch is named main

    batch_duration = time.time() - batch_start_time
    total_elapsed = time.time() - start_time
    print(f"⏱️ Batch duration: {timedelta(seconds=batch_duration)} | Total elapsed: {timedelta(seconds=total_elapsed)}")


print(f"{total_batches} processed.")

