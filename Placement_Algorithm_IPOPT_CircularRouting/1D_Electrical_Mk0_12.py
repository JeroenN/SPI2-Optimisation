import math

# Electrical Resistance
def electrical_resistance(resistivity, length, area):
    return resistivity * length / area

# Power Loss as Heat
def power_loss(current, resistance):
    return current ** 2 * resistance

# Voltage Drop
def voltage_drop(current, resistance):
    return current * resistance

# Heat Transfer (Fourier's Law)
def heat_transfer(thermal_conductivity, area, temp_gradient):
    return -thermal_conductivity * area * temp_gradient

# Efficiency
def efficiency(power_loss, power_input):
    if power_input == 0:
        return 0
    return 1 - (power_loss / power_input)

# Example usage
if __name__ == "__main__":
    # Example values
    ρ = 1.68e-8  # resistivity of copper in ohm meter           Fixed
    L = 2.0      # length in meters                             variable
    A = 1e-6     # cross-sectional area in m^2                  Fixed
    I = 5.0      # current in amperes                           Fixed
    k = 400      # thermal conductivity of copper in W/m·K      Fixed
    dT_dx = 50   # temperature gradient in K/m                  Fixed
    Pin = 100    # input power in watts                         Fixed

    R = electrical_resistance(ρ, L, A)
    Ploss = power_loss(I, R)
    Vdrop = voltage_drop(I, R)
    Q = heat_transfer(k, A, dT_dx)
    η = efficiency(Ploss, Pin)

    print(f"Resistance: {R:.6f} Ω")
    print(f"Power Loss: {Ploss:.6f} W")
    print(f"Voltage Drop: {Vdrop:.6f} V")
    print(f"Heat Transfer Rate: {Q:.6f} W")
    print(f"Efficiency: {η:.4%}")