import matplotlib.pyplot as plt
import pandas as pd
from sys import argv

if len(argv) != 2:
    print("Usage: {} path/to/energy.csv".format(sys.argv[0]))
    sys.exit(1)

fcsv = str(argv[1])
fpng = fcsv.replace("csv", "png")

data = pd.read_csv(fcsv)

plt.figure(figsize=(10,8))
plt.style.use("seaborn")
plt.title("FEniCS BM1b Free Energy")
plt.xlabel("Time (a.u.)")
plt.ylabel(u"Energy Density (J/m³)")

plt.semilogx(data["time"], data["free_energy"])
plt.savefig(fpng, dpi=400, bbox_inches="tight")
plt.close()
