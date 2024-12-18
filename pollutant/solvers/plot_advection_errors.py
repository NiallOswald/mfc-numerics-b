import numpy as np
import matplotlib.pyplot as plt

# Hard-code the data from the pollutant model
# This is done to avoid re-solving the model
esw = np.array(
    [
        [9.413e-11, 25000],  # 25k
        [1.0624e-10, 12500],  # 12_5k
        [1.1528e-10, 6250],  # 6_25k
    ]
)

las = np.array(
    [
        [2.789e-11, 40000],  # 40k
        [9.319e-11, 20000],  # 20k
        [1.0996e-10, 10000],  # 10k
        [1.1551e-10, 5000],  # 5k
        [1.1740e-10, 2500],  # 2_5k
        [1.1733e-10, 1250],  # 1_25k
    ]
)


def conv_rate(x):
    """Calculate the convergence rate of a sequence."""
    return np.log2(np.abs((x[1:-1] - x[:-2]) / (x[2:] - x[1:-1])))


def error(x):
    """Calculate the error of a sequence."""
    return np.abs(x[1:-1] - x[:-2]) / (1 - 2 ** (-conv_rate(x)))


# Plot the results
plt.figure()
plt.plot(esw[:, 1], esw[:, 0], "ko-", label="ESW")
plt.plot(las[:, 1], las[:, 0], "kx--", label="LAS")
plt.xlabel(r"Scale ($m$)")
plt.ylabel(r"Concentration ($m^{-2}$)")
plt.title(r"Convergence")
plt.legend()
plt.savefig("convergence.pdf", dpi=300)
plt.show()

# Print the convergence rates and errors
print("ESW Convergence rate:", conv_rate(esw[:, 0]))
print("LAS Convergence rate:", conv_rate(las[:, 0]))

print("ESW Error:", error(esw[:, 0]))
print("LAS Error:", error(las[:, 0]))

# Plot the errors
plt.figure()
# plt.loglog(esw[:-2, 1], error(esw[:, 0]), "ko-", label="ESW")  # Skip due to low data
plt.loglog(las[:-2, 1], error(las[:, 0]), "kx--", label="LAS")
# plot line of best fit
x = np.log(las[:-2, 1])
y = np.log(error(las[:, 0]))
m, c = np.polyfit(x, y, 1)
plt.loglog(las[:-2, 1], np.exp(m * x + c), "k:", label=f"$s = {m:.2f}$")

plt.xlabel(r"Scale ($m$)")
plt.ylabel(r"Error")
plt.title(r"Convergence in Absolute Error")
plt.legend()
plt.savefig("error.pdf", dpi=300)
plt.show()
