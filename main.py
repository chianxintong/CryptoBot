#import all necessary packages
import os
import telebot
import io
import time
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from PIL import Image
from telebot import types
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#set up telegram bot
API_KEY = os.environ['API_KEY']  # get api key from 'secrets' folder
bot = telebot.TeleBot(API_KEY)  # create bot

# Declare df_new as a global variable to access it in multiple functions
df_new = pd.DataFrame()


@bot.message_handler(commands=['start'])
def greet(message):
  bot.reply_to(
      message,
      "Hello, welcome to the crypto bot. To find out the top 10 cryptocurrencies, simply type /crypto"
  )


#use beautiful soup to scrape yahoo finance website and extract relevant forex data
def get_webpage(url):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, "html.parser")
  return soup


#convert scraped forex data to pandas dataframe
def get_curr_df(soup):
  rows = soup.find("table").find_all("tr")
  data = []
  for row in rows:
    cells = row.find_all("td")[0:5]
    data.append([cell.text for cell in cells])
  df = pd.DataFrame(data,
                    columns=['symbol', 'name', 'price', 'change', '% change'
                             ]).drop(0, axis=0)  # drop first row (ie. titles)
  return df


#adjust formatting of data and perform data cleaning
def clean_data(df):
  # Remove commas from the 'change' column
  df['change'] = df['change'].str.replace(',', '')

  # Convert to numeric data type for ease of comparison
  df['change'] = pd.to_numeric(df['change'], errors='coerce')
  df['price'] = df['price'].str.replace(',', '').astype(float)
  df['% change'] = df['% change'].str.rstrip("%")
  df['% change'] = pd.to_numeric(df['% change'], errors='coerce')
  return df


#create a bar chart featuring top_n forex pairs
def generate_image(dataframe, top_n):
  fig, ax = plt.subplots(figsize=(8, 10))
  ax.set_title('Top {} cryptocurrencies'.format(top_n))
  #create seaborn bar plot with dataframe provided
  sns.barplot(x='symbol', y='% change', data=dataframe, ax=ax)
  #specify axis labels
  ax.set(xlabel='cryptocurrency', ylabel='% change')
  #rotate tick labels on the x axis to ensure visibility
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
  ax.bar_label(ax.containers[0], fontsize=10)
  buf = io.BytesIO()
  # save buffered image
  fig.savefig(buf, format='png')
  # use PIL image to derive image object
  buf.seek(0)
  img = Image.open(buf)
  return img


def create_cryptodf(symbol):
  ticker = yf.Ticker(symbol)
  end_date=datetime.now().strftime('%Y-%m-%d')
  start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
  df = ticker.history(start=start_date,
                    end=end_date,
                    interval="1h")
  df.reset_index(inplace=True)
  df.rename(columns={'Datetime': 'Date'}, inplace=True)
  return df


def forecast_image(crypto_df):
  # Exponential smoothing model
  model = ExponentialSmoothing(crypto_df['Close'],
                               trend='add',
                               seasonal='add',
                               seasonal_periods=12)
  fit = model.fit()
  # Forecasting future values
  forecast_steps = 10  # Adjust the number of steps as needed
  forecast_values = fit.forecast(steps=forecast_steps)
  # Plotting the graph
  fig, ax = plt.subplots(figsize=(12, 6))
  # Blue part: Past trend
  plt.plot(crypto_df['Date'],
           crypto_df['Close'],
           label='Past Trend',
           color='blue')
  # Red part: Forecasted values
  forecast_dates = pd.date_range(crypto_df['Date'].max(),
                                 periods=forecast_steps + 1,
                                 freq='D')[1:]
  plt.plot(forecast_dates,
           forecast_values,
           label='Forecasted Values',
           color='red',
           linestyle='dashed')
  plt.title('Cryptocurrency Price Forecast')
  plt.xlabel('Date')
  plt.ylabel('Close Price')
  plt.legend()
  buf = io.BytesIO()
  # save buffered image
  fig.savefig(buf, format='png')
  # use PIL image to derive image object
  buf.seek(0)
  img = Image.open(buf)
  return img


#generate bar chart of top 10 crypto currencies and send image to the chat
@bot.message_handler(commands=['crypto'])
def send_photo(message):
  global df_new
  soup = get_webpage('https://sg.finance.yahoo.com/crypto/')
  df = get_curr_df(soup)
  clean_data(df)
  sorted_df = df.sort_values(by=['% change'], ascending=False)
  df_new = sorted_df.head(10)
  bot.send_chat_action(message.chat.id, 'upload_photo')
  img = generate_image(df_new, 10)

  # addition of button to navigate to top 10 cryto forecasted rates
  markup = types.InlineKeyboardMarkup()
  button = types.InlineKeyboardButton(text="Check forecasted rates",
                                      callback_data="forecast")
  markup.add(button)
  bot.send_photo(message.chat.id,
                 img,
                 reply_to_message_id=message.message_id,
                 reply_markup=markup)  #reply_markup is for addition of button
  img.close()


# Creates buttons for each crypto that leads to indiv forecasted rates
@bot.callback_query_handler(func=lambda call: call.data == "forecast")
def handle_forecast_query(call):
  symbols = df_new['symbol'].tolist()  # Extract symbols from df as a list
  markup = types.InlineKeyboardMarkup()
  for symbol in symbols:
    button = types.InlineKeyboardButton(text=symbol, callback_data=symbol)
    markup.add(button)
  bot.send_message(chat_id=call.message.chat.id,
                   text='Choose a cryptocurrency:',
                   reply_markup=markup)


@bot.callback_query_handler(
    func=lambda call: call.data in df_new['symbol'].tolist())
def handle_symbol_query(call):
  symbol = call.data
  crypto_df = create_cryptodf(symbol)
  bot.send_chat_action(call.message.chat.id, 'upload_photo')
  img = forecast_image(crypto_df)
  bot.send_photo(call.message.chat.id,
                 img,
                 reply_to_message_id=call.message.message_id)
  img.close()

bot.polling()  # keeps checking for msgs
plt.show()
while True:
  time.sleep(0)
