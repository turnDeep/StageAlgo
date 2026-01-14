import trendln
import numpy as np

h = np.random.randn(200).cumsum()

try:
    print("Testing with accuracy=2...")
    ret = trendln.calc_support_resistance(h, window=50, accuracy=2)
    print("Return type:", type(ret))
    print("Return length:", len(ret))
    for i, item in enumerate(ret):
        print(f"Item {i} type: {type(item)}")
        # print first few elements
        try:
             print(f"Item {i}: {item[:2]}")
        except:
             print(f"Item {i}: {item}")

except Exception as e:
    print("Error with acc=2:", e)
