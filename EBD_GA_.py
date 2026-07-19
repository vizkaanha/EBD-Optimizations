import random
import time

import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools

# ==========================================================
# REPRODUCIBILITY AND TIMER
# ==========================================================
SEED = 21
random.seed(SEED)
rng = np.random.default_rng(SEED)

start_time = time.time()
plt.rcParams["figure.figsize"] = (7, 5)

# ==========================================================
# SYSTEM PARAMETERS
# ==========================================================
Z0 = 50.0
X = np.arange(0, 101, dtype=int)

# Requested EBD SIC design range.
SIC_MIN_DB = 40.0
SIC_MAX_DB = 80.0

# An 80 dB SIC ceiling corresponds to an amplitude-leakage floor of 1e-4.
# The quadrature addition below avoids artificial values above the model floor.
LEAKAGE_FLOOR = 10.0 ** (-SIC_MAX_DB / 20.0)
LEAKAGE_AT_40_DB = 10.0 ** (-SIC_MIN_DB / 20.0)

# Practical residual imbalance used for robust GA evaluation.
# These are simulation assumptions, not universal EBD specifications.
AMP_ERROR_STD = 0.002          
PHASE_ERROR_STD_DEG = 0.15     
NOISE_STD = 2e-5               
MONTE_CARLO_SAMPLES = 24

# GA settings.
POP = 60
GEN = 50
CXPB = 0.70
MUTPB = 0.25

# ==========================================================
# STORAGE LISTS
# ==========================================================
Lb = []
La = []
Sb = []
Sa = []
Vb = []
Va = []
Xopt = []
Stat = []

# ==========================================================
# CREATE DEAP CLASSES ONCE
# ==========================================================
if not hasattr(creator, "FitnessMaxEBD"):
    creator.create("FitnessMaxEBD", base.Fitness, weights=(1.0,))

if not hasattr(creator, "IndividualEBD"):
    creator.create(
        "IndividualEBD",
        list,
        fitness=creator.FitnessMaxEBD,
    )


def zlbl(x_reactance):
    """Return the displayed antenna impedance label."""
    return f"50+j{int(x_reactance)} ohm"


def sstat(sic_db):
    """Classify the optimized SIC against the requested design range."""
    if sic_db < SIC_MIN_DB:
        return "LOW"
    if sic_db < SIC_MAX_DB:
        return "PASS"
    return "AT LIMIT"


# ==========================================================
# MAIN ANTENNA-IMPEDANCE LOOP
# ==========================================================
for Xa in X:
    Zant = Z0 + 1j * Xa
    Ga = (Zant - Z0) / (Zant + Z0)

    amp_samples = rng.normal(1.0, AMP_ERROR_STD, MONTE_CARLO_SAMPLES)
    phase_samples = np.deg2rad(
        rng.normal(0.0, PHASE_ERROR_STD_DEG, MONTE_CARLO_SAMPLES)
    )
    noise_samples = (
        rng.normal(0.0, NOISE_STD, MONTE_CARLO_SAMPLES)
        + 1j * rng.normal(0.0, NOISE_STD, MONTE_CARLO_SAMPLES)
    )

    # ======================================================
    # BEFORE OPTIMIZATION
    # ======================================================
    Zbal0 = Z0 + 0j
    Gamma_bal0 = (Zbal0 - Z0) / (Zbal0 + Z0)

    Vrx0_samples = (
        Ga
        + amp_samples
        * Gamma_bal0
        * np.exp(-1j * (np.pi + phase_samples))
        + noise_samples
    )

    # RMS leakage with an explicit measurement/model floor.
    leak0 = np.sqrt(np.mean(np.abs(Vrx0_samples) ** 2) + LEAKAGE_FLOOR**2)
    sic0 = -20.0 * np.log10(leak0)

    Lb.append(float(leak0))
    Sb.append(float(sic0))

    gamma_before = np.clip(np.abs(Ga), 0.0, 0.999999)
    Vb.append(float((1.0 + gamma_before) / (1.0 - gamma_before)))

    # ======================================================
    # GENETIC ALGORITHM
    # ======================================================
    toolbox = base.Toolbox()
    toolbox.register("gene", random.uniform, 0.0, 100.0)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.IndividualEBD,
        toolbox.gene,
        n=1,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_candidate(individual):
        """Robustly score one integer balancing-reactance candidate."""
        Xbal = int(np.clip(np.rint(individual[0]), 0, 100))
        Zbal = Z0 + 1j * Xbal
        Gb = (Zbal - Z0) / (Zbal + Z0)

        Vrx_samples = (
            Ga
            + amp_samples
            * Gb
            * np.exp(-1j * (np.pi + phase_samples))
            + noise_samples
        )

        effective_leakage = np.sqrt(
            np.abs(Vrx_samples) ** 2 + LEAKAGE_FLOOR**2
        )
        sic_samples = -20.0 * np.log10(effective_leakage)

        # A lower-percentile value makes the optimization robust rather
        # than rewarding only an unusually favourable random sample.
        robust_sic = float(np.percentile(sic_samples, 10))
        mean_sic = float(np.mean(sic_samples))

        # Reward high robust SIC inside the allowed model range.
        # Strongly penalize candidates that fail the 40 dB minimum.
        if robust_sic < SIC_MIN_DB:
            score = robust_sic - 8.0 * (SIC_MIN_DB - robust_sic)
        else:
            score = 0.75 * robust_sic + 0.25 * mean_sic

        return (score,)

    toolbox.register("evaluate", evaluate_candidate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register(
        "mutate",
        tools.mutGaussian,
        mu=0.0,
        sigma=5.0,
        indpb=0.5,
    )
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POP)
    hall_of_fame = tools.HallOfFame(1)

    # Evaluate the initial population.
    invalid_individuals = [ind for ind in population if not ind.fitness.valid]
    initial_fitnesses = map(toolbox.evaluate, invalid_individuals)
    for individual, fitness_value in zip(invalid_individuals, initial_fitnesses):
        individual.fitness.values = fitness_value
    hall_of_fame.update(population)

    # Generational evolution with one elite individual retained.
    for _ in range(GEN):
        elite = tools.selBest(population, 1)[0]
        offspring = toolbox.select(population, len(population) - 1)
        offspring = list(map(toolbox.clone, offspring))
        offspring = algorithms.varAnd(
            offspring,
            toolbox,
            cxpb=CXPB,
            mutpb=MUTPB,
        )

        invalid_individuals = [ind for ind in offspring if not ind.fitness.valid]
        offspring_fitnesses = map(toolbox.evaluate, invalid_individuals)
        for individual, fitness_value in zip(
            invalid_individuals, offspring_fitnesses
        ):
            individual.fitness.values = fitness_value

        population = offspring + [toolbox.clone(elite)]
        hall_of_fame.update(population)

    best_individual = hall_of_fame[0]
    xb = int(np.clip(np.rint(best_individual[0]), 0, 100))
    Xopt.append(xb)

    # ======================================================
    # AFTER OPTIMIZATION
    # ======================================================
    Zbal = Z0 + 1j * xb
    Gb = (Zbal - Z0) / (Zbal + Z0)

    Vrx_samples = (
        Ga
        + amp_samples
        * Gb
        * np.exp(-1j * (np.pi + phase_samples))
        + noise_samples
    )

    leak = np.sqrt(np.mean(np.abs(Vrx_samples) ** 2) + LEAKAGE_FLOOR**2)
    sic = -20.0 * np.log10(leak)

    La.append(float(leak))
    Sa.append(float(sic))
    Stat.append(sstat(float(sic)))

    # This is an equivalent residual-mismatch ratio derived from leakage;
    # the GA tunes the balance network and does not physically retune Zant.
    gamma_residual = np.clip(leak, 0.0, 0.999999)
    Va.append(float((1.0 + gamma_residual) / (1.0 - gamma_residual)))

    print(f"Zant = {zlbl(Xa)}")
    print(f"Optimized balance reactance Xbal = {xb} ohm")
    print(f"Robust GA fitness = {best_individual.fitness.values[0]:.3f}")
    print(f"SIC after GA = {sic:.2f} dB ({sstat(sic)})")
    print(f"SIC improvement = {sic - sic0:.2f} dB")
    print("-" * 66)

# ==========================================================
# CONVERT TO NUMPY ARRAYS
# ==========================================================
Lb = np.asarray(Lb, dtype=float)
La = np.asarray(La, dtype=float)
Sb = np.asarray(Sb, dtype=float)
Sa = np.asarray(Sa, dtype=float)
Vb = np.asarray(Vb, dtype=float)
Va = np.asarray(Va, dtype=float)
Xopt = np.asarray(Xopt, dtype=int)

end_time = time.time()
print(f"\nTotal execution time = {end_time - start_time:.3f} seconds")

# ==========================================================
# CREATE ALL PLOTS IN ONE FIGURE
# ==========================================================

# Separate figures (all open simultaneously)
impedance_tick_positions = np.arange(0, 101, 20)
impedance_tick_labels = [f"50+j{x}" for x in impedance_tick_positions]

# Figure 1
plt.figure(figsize=(7,5))
plt.plot(X, Lb, linewidth=2.5, label="Before GA")
plt.plot(X, La, linewidth=2.5, label="After GA")
plt.axhspan(LEAKAGE_FLOOR, LEAKAGE_AT_40_DB, alpha=0.15,
            label="40-80 dB SIC region")
plt.axhline(LEAKAGE_AT_40_DB, linestyle="--")
plt.axhline(LEAKAGE_FLOOR, linestyle="--")
plt.yscale("log")
plt.xlabel(r"Antenna Impedance $Z_{ant}=50+jX$ ($\Omega$)")
plt.ylabel("RMS residual leakage amplitude")
plt.title("Leakage Reduction using Robust GA")
plt.legend()
plt.grid(True, which="both")
plt.xticks(impedance_tick_positions, impedance_tick_labels,
           rotation=30)

# Figure 2
plt.figure(figsize=(7,5))
plt.plot(X, Sb, linewidth=2.5, label="Before GA")
plt.plot(X, Sa, linewidth=2.5, label="After GA")
plt.axhspan(SIC_MIN_DB, SIC_MAX_DB, alpha=0.15,
            label="Target: 40-80 dB")
plt.axhline(SIC_MIN_DB, linestyle="--")
plt.axhline(SIC_MAX_DB, linestyle="--")
plt.xlabel(r"Antenna Impedance $Z_{ant}=50+jX$ ($\Omega$)")
plt.ylabel("SIC (dB)")
plt.title("SIC with 40-80 dB Design Limits")
plt.legend()
plt.grid(True)
plt.xticks(impedance_tick_positions, impedance_tick_labels,
           rotation=30)

# Figure 3
plt.figure(figsize=(7,5))
plt.plot(X, Vb, linewidth=2.5, label="Antenna VSWR before")
plt.plot(X, Va, linewidth=2.5,
         label="Equivalent residual VSWR after GA")
plt.xlabel(r"Antenna Impedance $Z_{ant}=50+jX$ ($\Omega$)")
plt.ylabel("VSWR")
plt.title("Antenna and Residual-Mismatch VSWR")
plt.legend()
plt.grid(True)
plt.xticks(impedance_tick_positions, impedance_tick_labels,
           rotation=30)

plt.show()
# ==========================================================
# FORMATTED TABLE
# ==========================================================
table_width = 139
print("\n" + "=" * table_width)
print(
    f"{'S.No.':<8}"
    f"{'Antenna Impedance':<24}"
    f"{'VSWR Before':<17}"
    f"{'Residual VSWR':<17}"
    f"{'Leak Before':<18}"
    f"{'Leak After':<18}"
    f"{'SIC Before':<16}"
    f"{'SIC After':<16}"
)
print("-" * table_width)

for i in range(len(X)):
    print(
        f"{i + 1:<8}"
        f"{zlbl(X[i]):<24}"
        f"{Vb[i]:<17.3f}"
        f"{Va[i]:<17.3f}"
        f"{Lb[i]:<18.5e}"
        f"{La[i]:<18.5e}"
        f"{Sb[i]:<16.2f}"
        f"{Sa[i]:<16.2f}"
    )

print("=" * table_width)

# ==========================================================
# SIC RANGE SUMMARY
# ==========================================================
before_min_index = int(np.argmin(Sb))
before_max_index = int(np.argmax(Sb))
after_min_index = int(np.argmin(Sa))
after_max_index = int(np.argmax(Sa))
pass_count = int(np.count_nonzero((Sa >= SIC_MIN_DB) & (Sa < SIC_MAX_DB)))
low_count = int(np.count_nonzero(Sa < SIC_MIN_DB))
at_limit_count = int(np.count_nonzero(Sa >= SIC_MAX_DB))

print("\n" + "=" * 88)
print("SIC TARGET AND MINIMUM/MAXIMUM SUMMARY")
print("=" * 88)
print(f"Requested SIC design range: {SIC_MIN_DB:.0f} dB <= SIC < {SIC_MAX_DB:.0f} dB")
print(
    f"Before GA minimum SIC = {Sb[before_min_index]:.2f} dB "
    f"at Zant = {zlbl(X[before_min_index])}"
)
print(
    f"Before GA maximum SIC = {Sb[before_max_index]:.2f} dB "
    f"at Zant = {zlbl(X[before_max_index])}"
)
print(
    f"After GA minimum SIC  = {Sa[after_min_index]:.2f} dB "
    f"at Zant = {zlbl(X[after_min_index])}"
)
print(
    f"After GA maximum SIC  = {Sa[after_max_index]:.2f} dB "
    f"at Zant = {zlbl(X[after_max_index])}"
)
print(f"PASS points (40-80 dB): {pass_count}/{len(X)}")
print(f"LOW points (<40 dB):    {low_count}/{len(X)}")
print(f"AT/ABOVE LIMIT points:  {at_limit_count}/{len(X)}")
print("=" * 88)
