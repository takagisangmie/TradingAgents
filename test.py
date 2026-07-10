import time

from tradingagents.dataflows.tushare import get_indicators

print("Testing optimized implementation with 30-day lookback:")
start_time = time.time()
result = get_indicators("600519.SH", "macd", "2026-01-15", 30)
end_time = time.time()

print(f"Execution time: {end_time - start_time:.2f} seconds")
print(f"Result length: {len(result)} characters")
print(result)
