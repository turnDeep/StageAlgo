import trendln
import inspect

print("trendln functions:")
print(dir(trendln))

print("\ncalc_support_resistance signature:")
try:
    print(inspect.signature(trendln.calc_support_resistance))
except:
    print("Could not get signature")

print("\nplot_support_resistance signature:")
try:
    print(inspect.signature(trendln.plot_support_resistance))
except:
    print("Could not get signature")
