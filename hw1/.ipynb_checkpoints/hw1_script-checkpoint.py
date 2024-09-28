import pandas as pd
import yfinance as yf
from summarytools import dfSummary
import matplotlib.pyplot as plt
    

def main():
    tickers = ['^GSPC', 'BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']
    df = yf.download(tickers, period="1mo", interval='1d')['Close']
    
    df.dropna(inplace=True)
    
    print(dfSummary(df))

    for ticker in tickers:
        plt.figure()  
        df[ticker].plot(title=ticker)
        plt.show()
    
    

if __name__ == "__main__":
    main()