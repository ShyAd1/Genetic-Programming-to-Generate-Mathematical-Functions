
import operator
import math
import random
import datetime
from pathlib import Path
from functools import partial

import numpy
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx

from deap import base, creator, tools, gp, algorithms

# ══════════════════════════════════════════════════════════════
# SECCIÓN 1 — CONFIGURACIÓN GLOBAL (editar aquí)
# ══════════════════════════════════════════════════════════════

# Función objetivo como cadena de texto (usa x e y como variables)
TARGET_EXPR_STR = "x**2 + y**2*x**2 + 1"

# Parámetros del algoritmo
POPULATION_SIZE      = 300
N_GENERATIONS        = 100
CROSSOVER_PROB       = 0.80
MUTATION_PROB        = 0.10
TREE_MIN_DEPTH       = 1
TREE_MAX_DEPTH       = 6
TREE_HEIGHT_LIMIT    = 17

# Penalización por complejidad: fitness = MSE + COMPLEXITY_WEIGHT * len(árbol)
COMPLEXITY_WEIGHT    = 0.005

# Parada anticipada: detiene la evolución si MSE < umbral
EARLY_STOP_THRESHOLD = 1e-4

# Puntos de evaluación: cuadrícula [-5, 5] x [-5, 5]
EVAL_RANGE           = range(-50, 51)  # puntos cada 0.1 en [-5, 5]
EVAL_POINTS          = [
    (x / 10.0, y / 10.0) for x in EVAL_RANGE for y in EVAL_RANGE
]

# Semilla para reproducibilidad
RANDOM_SEED          = 42  # random.randint(0, 10000)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 2 — FUNCIÓN OBJETIVO DINÁMICA
# ══════════════════════════════════════════════════════════════

def build_target_function(expr_str: str):
    """
    Convierte una cadena de texto en una función Python evaluable.
    Usa SymPy para parsear y lambdify para compilar.
    """
    x_sym, y_sym = sp.symbols("x y")
    try:
        sympy_expr = sp.sympify(expr_str)
        lambdified = sp.lambdify((x_sym, y_sym), sympy_expr, "numpy")
        print(f"✔ Función objetivo cargada: f(x,y) = {sympy_expr}")
        return lambdified
    except Exception as e:
        raise ValueError(f"No se pudo parsear la función objetivo '{expr_str}': {e}")


_target_fn = build_target_function(TARGET_EXPR_STR)

def target_function(x, y):
    return _target_fn(x, y)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 3 — OPERADORES PROTEGIDOS EXTENDIDOS
# ══════════════════════════════════════════════════════════════

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1.0

def protectedSqrt(x):
    return math.sqrt(abs(x))

def protectedLog(x):
    return math.log(abs(x)) if abs(x) > 1e-10 else 0.0

def protectedExp(x):
    try:
        result = math.exp(min(x, 700))   # evita overflow
        return result if math.isfinite(result) else 1.0
    except OverflowError:
        return 1.0

def protectedAbs(x):
    return abs(x)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 4 — PRIMITIVOS Y CREADORES DEAP
# ══════════════════════════════════════════════════════════════

pset = gp.PrimitiveSet("MAIN", 2)

# Operadores aritméticos
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

# Operadores extendidos
# pset.addPrimitive(math.sin,     1)
# pset.addPrimitive(math.cos,     1)
# pset.addPrimitive(protectedSqrt, 1)
# pset.addPrimitive(protectedLog,  1)
# pset.addPrimitive(protectedExp,  1)
# pset.addPrimitive(protectedAbs,  1)

pset.addEphemeralConstant("rand101", partial(random.randint, -5, 5))
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset,
                 min_=TREE_MIN_DEPTH, max_=TREE_MAX_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
toolbox.register("compile",     gp.compile, pset=pset)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 5 — FUNCIÓN DE EVALUACIÓN CON PENALIZACIÓN
# ══════════════════════════════════════════════════════════════

def evalSymbReg(individual, points):
    """
    Fitness = MSE + COMPLEXITY_WEIGHT * tamaño_árbol

    La penalización por complejidad desincentiva árboles muy grandes
    (bloat) sin comprometer demasiado la precisión.
    """
    func = toolbox.compile(expr=individual)
    PENALTY = 1e20
    sqerrors = []

    for x, y in points:
        try:
            pred  = func(x, y)
            error = pred - target_function(x, y)
            sq    = error * error
        except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError):
            return (PENALTY,)

        if not math.isfinite(sq):
            return (PENALTY,)

        sqerrors.append(sq)

    mse              = math.fsum(sqerrors) / len(points)
    complexity_bonus = COMPLEXITY_WEIGHT * len(individual)
    return (mse + complexity_bonus,)


toolbox.register("evaluate", evalSymbReg, points=EVAL_POINTS)
toolbox.register("select",   tools.selTournament, tournsize=3)
toolbox.register("mate",     gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate",   gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"),
                                          max_value=TREE_HEIGHT_LIMIT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"),
                                          max_value=TREE_HEIGHT_LIMIT))


# ══════════════════════════════════════════════════════════════
# SECCIÓN 6 — BUCLE EVOLUTIVO CON PARADA ANTICIPADA
# ══════════════════════════════════════════════════════════════

def run_evolution(pop, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose=True):
    """
    Bucle evolutivo personalizado con:
      - Parada anticipada cuando MSE < EARLY_STOP_THRESHOLD
      - Registro del mejor individuo por generación
      - Cálculo de diversidad genética (fracción de árboles únicos)
    """
    logbook          = tools.Logbook()
    logbook.header   = ["gen", "nevals"] + (stats.fields if stats else [])

    best_per_gen     = []   # Mejor individuo clonado por generación
    diversity_per_gen = []  # Proporción de individuos únicos

    # ── Generación 0: evaluar población inicial ──────────────
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(pop)

    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid), **record)
    if verbose:
        print(logbook.stream)

    best_per_gen.append(toolbox.clone(halloffame[0]))
    diversity_per_gen.append(len(set(str(i) for i in pop)) / len(pop))

    early_stopped = False

    # ── Generaciones 1..ngen ─────────────────────────────────
    for gen in range(1, ngen + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # Cruza
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutación
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluar
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring

        if halloffame is not None:
            halloffame.update(pop)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

        best_per_gen.append(toolbox.clone(halloffame[0]))
        diversity_per_gen.append(len(set(str(i) for i in pop)) / len(pop))

        # ── Parada anticipada ────────────────────────────────
        current_best = halloffame[0].fitness.values[0]
        if current_best < EARLY_STOP_THRESHOLD:
            print(f"\n⏹  Parada anticipada en generación {gen}: "
                  f"fitness = {current_best:.2e} < {EARLY_STOP_THRESHOLD:.2e}")
            early_stopped = True
            break

    return pop, logbook, best_per_gen, diversity_per_gen, early_stopped


# ══════════════════════════════════════════════════════════════
# SECCIÓN 7 — SIMPLIFICACIÓN SIMBÓLICA (SymPy)
# ══════════════════════════════════════════════════════════════

_OP_MAP = {
    "add":          lambda a, b: a + b,
    "sub":          lambda a, b: a - b,
    "mul":          lambda a, b: a * b,
    "protectedDiv": lambda a, b: a / b,
    "neg":          lambda a:    -a,
    "sin":          lambda a:    sp.sin(a),
    "cos":          lambda a:    sp.cos(a),
    "protectedSqrt": lambda a:  sp.sqrt(sp.Abs(a)),
    "protectedLog":  lambda a:  sp.log(sp.Abs(a) + sp.Float(1e-10)),
    "protectedExp":  lambda a:  sp.exp(a),
    "protectedAbs":  lambda a:  sp.Abs(a),
}

def simplify_best_individual(individual):
    """Convierte el árbol DEAP a expresión SymPy y la simplifica."""
    x_sym, y_sym = sp.symbols("x y")

    def _to_sympy(tree, idx=0):
        if idx >= len(tree):
            return None, idx
        node = tree[idx]

        # Terminal
        if node.arity == 0:
            if node.name in ("x", "ARG0"):
                return x_sym, idx + 1
            if node.name in ("y", "ARG1"):
                return y_sym, idx + 1
            try:
                return sp.Integer(int(node.value)), idx + 1
            except Exception:
                return sp.symbols(node.name), idx + 1

        # Nodo interno
        args, next_idx = [], idx + 1
        for _ in range(node.arity):
            arg, next_idx = _to_sympy(tree, next_idx)
            if arg is not None:
                args.append(arg)

        fn = _OP_MAP.get(node.name)
        try:
            result = fn(*args) if fn else (args[0] if args else None)
        except Exception:
            result = args[0] if args else None

        return result, next_idx

    try:
        expr, _ = _to_sympy(individual)
        if expr is None:
            return None
        simplified = sp.simplify(expr)
        expanded   = sp.expand(simplified)
        return expanded if str(expanded) != str(simplified) else simplified
    except Exception as e:
        print(f"Error al simplificar: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# SECCIÓN 8 — MÉTRICAS ADICIONALES
# ══════════════════════════════════════════════════════════════

def compute_metrics(individual, points):
    """Calcula MSE, RMSE, MAE y R² para el mejor individuo."""
    func = toolbox.compile(expr=individual)
    sq_errors, abs_errors, y_true, y_pred = [], [], [], []

    for x, y in points:
        try:
            pred = func(x, y)
            true = target_function(x, y)
            if math.isfinite(pred) and math.isfinite(true):
                sq_errors.append((pred - true) ** 2)
                abs_errors.append(abs(pred - true))
                y_true.append(true)
                y_pred.append(pred)
        except Exception:
            pass

    if not sq_errors:
        return None

    y_true_arr = numpy.array(y_true)
    mse   = numpy.mean(sq_errors)
    mae   = numpy.mean(abs_errors)
    ss_res = numpy.sum(sq_errors)
    ss_tot = numpy.sum((y_true_arr - y_true_arr.mean()) ** 2)
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "MSE":  mse,
        "RMSE": math.sqrt(mse),
        "MAE":  mae,
        "R²":   r2,
    }


# ══════════════════════════════════════════════════════════════
# SECCIÓN 9 — VISUALIZACIÓN DEL ÁRBOL DE EXPRESIÓN
# ══════════════════════════════════════════════════════════════

def _hierarchy_pos(G, root, width=1.0, vert_gap=0.25, vert_loc=0.0,
                   xcenter=0.5, pos=None):
    """Layout jerárquico sin dependencia de Graphviz."""
    if pos is None:
        pos = {}
    pos[root] = (xcenter, vert_loc)
    children = list(G.successors(root))
    if children:
        dx = width / len(children)
        x0 = xcenter - width / 2 + dx / 2
        for i, child in enumerate(children):
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                 vert_loc=vert_loc - vert_gap,
                                 xcenter=x0 + i * dx, pos=pos)
    return pos


def plot_expression_tree(individual, output_dir, filename="arbol_expresion.png"):
    """Dibuja el árbol de expresión GP usando NetworkX."""
    nodes, edges, labels = gp.graph(individual)

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    root = nodes[0] if nodes else 0
    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = _hierarchy_pos(g, root)

    # Colorear nodos según tipo
    node_colors = []
    for n in g.nodes():
        lbl = str(labels.get(n, ""))
        try:
            int(lbl)
            node_colors.append("#27ae60")   # constante → verde
        except ValueError:
            if lbl in ("x", "y"):
                node_colors.append("#2ecc71")   # variable → verde claro
            else:
                node_colors.append("#2980b9")   # operador → azul

    fig, ax = plt.subplots(figsize=(max(12, len(nodes) * 0.6), 7))
    nx.draw(
        g, pos, ax=ax, labels=labels, with_labels=True,
        node_color=node_colors, node_size=900,
        font_size=8, font_color="white", font_weight="bold",
        arrows=True, arrowsize=12, edge_color="#7f8c8d", width=1.5,
    )

    # Leyenda
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2980b9", label="Operador"),
        Patch(facecolor="#2ecc71", label="Variable"),
        Patch(facecolor="#27ae60", label="Constante"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    ax.set_title(
        f"Árbol de expresión — Mejor individuo\n"
        f"Nodos: {len(individual)}  |  Profundidad: {individual.height}",
        fontsize=13,
    )
    plt.tight_layout()
    path = output_dir / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada: {path}")
    return path


# ══════════════════════════════════════════════════════════════
# SECCIÓN 10 — GRÁFICAS DE ANÁLISIS
# ══════════════════════════════════════════════════════════════

def show_plots(log, hof, best_per_gen, diversity_per_gen, output_dir):
    generations  = log.select("gen")
    min_fitness  = log.chapters["fitness"].select("min")
    avg_fitness  = log.chapters["fitness"].select("avg")
    avg_size     = log.chapters["size"].select("avg")
    max_size     = log.chapters["size"].select("max")
    avg_depth    = log.chapters["depth"].select("avg")
    max_depth    = log.chapters["depth"].select("max")

    best_fitness_history = [ind.fitness.values[0] for ind in best_per_gen]
    gens_plot = generations[: len(diversity_per_gen)]

    # ── Panel principal: 2×3 subplots ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Análisis Evolutivo — Regresión Simbólica DEAP",
                 fontsize=16, fontweight="bold")

    # 1. Evolución del MSE (escala log)
    ax = axes[0, 0]
    ax.semilogy(generations, min_fitness, "o-", ms=3,
                color="#e74c3c", label="MSE mínimo")
    ax.semilogy(generations, avg_fitness, "s-", ms=3,
                color="#3498db", alpha=0.6, label="MSE promedio")
    ax.set_title("Evolución del Error (MSE)")
    ax.set_xlabel("Generación"); ax.set_ylabel("MSE (log)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Fitness del mejor individuo
    ax = axes[0, 1]
    ax.semilogy(gens_plot, best_fitness_history, "o-", ms=3, color="#1abc9c")
    ax.axhline(EARLY_STOP_THRESHOLD, color="red", ls="--",
               lw=1, label=f"Umbral: {EARLY_STOP_THRESHOLD:.0e}")
    ax.set_title("Fitness del Mejor Individuo")
    ax.set_xlabel("Generación"); ax.set_ylabel("Fitness (log)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Diversidad genética
    ax = axes[0, 2]
    ax.plot(gens_plot, diversity_per_gen, "o-", ms=3, color="#9b59b6")
    ax.set_title("Diversidad Genética")
    ax.set_xlabel("Generación"); ax.set_ylabel("Fracción de árboles únicos")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # 4. Tamaño de individuos (complejidad)
    ax = axes[1, 0]
    ax.plot(generations, avg_size, "o-", ms=3, color="#2ecc71", label="Promedio")
    ax.plot(generations, max_size, "s-", ms=3, color="#e67e22", alpha=0.6, label="Máximo")
    ax.set_title("Complejidad — Nodos por Árbol")
    ax.set_xlabel("Generación"); ax.set_ylabel("Número de nodos")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Profundidad de árboles
    ax = axes[1, 1]
    ax.plot(generations, avg_depth, "o-", ms=3, color="#16a085", label="Promedio")
    ax.plot(generations, max_depth, "s-", ms=3, color="#c0392b", alpha=0.6, label="Máxima")
    ax.set_title("Profundidad de los Árboles")
    ax.set_xlabel("Generación"); ax.set_ylabel("Profundidad")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Tamaño del mejor individuo por generación
    best_sizes = [len(ind) for ind in best_per_gen]
    ax = axes[1, 2]
    ax.plot(gens_plot, best_sizes, "o-", ms=3, color="#e67e22")
    ax.set_title("Tamaño del Mejor Individuo")
    ax.set_xlabel("Generación"); ax.set_ylabel("Nodos")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    evo_path = output_dir / "evolucion_estadisticas.png"
    plt.savefig(evo_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada: {evo_path}")

    # ── Superficies 3D: real | aproximada | error absoluto ──────
    best_func = toolbox.compile(expr=hof[0])
    x = numpy.linspace(-1, 1, 40)
    y = numpy.linspace(-1, 1, 40)
    X, Y = numpy.meshgrid(x, y)

    Z_real = target_function(X, Y)
    Z_pred = numpy.array([
        [best_func(xi, yi) for xi, yi in zip(rx, ry)]
        for rx, ry in zip(X, Y)
    ])
    Z_err = numpy.abs(Z_real - Z_pred)

    fig = plt.figure(figsize=(18, 5))
    titles   = ["Función objetivo", "Mejor aproximación DEAP", "Error absoluto |f − f̂|"]
    Z_list   = [Z_real, Z_pred, Z_err]
    cmaps    = ["viridis", "plasma", "hot"]

    for k, (title, Z, cmap) in enumerate(zip(titles, Z_list, cmaps)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor="none")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    plt.suptitle("Comparación de Superficies 3D", fontsize=14, fontweight="bold")
    plt.tight_layout()
    surf_path = output_dir / "superficies_deap.png"
    plt.savefig(surf_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada: {surf_path}")

    return evo_path, surf_path


# ══════════════════════════════════════════════════════════════
# SECCIÓN 11 — GUARDADO DE RESULTADOS
# ══════════════════════════════════════════════════════════════

def save_results(hof, simplified_expr, metrics, log, output_dir,
                 early_stopped, best_per_gen):
    """Escribe un resumen completo de la ejecución en un archivo .txt."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / "Resultados" / f"resultados_{timestamp}.txt"

    gens     = log.select("gen")
    mins     = log.chapters["fitness"].select("min")
    avg_size = log.chapters["size"].select("avg")

    with open(path, "w", encoding="utf-8") as f:
        sep  = "═" * 60
        dash = "─" * 40

        f.write(f"{sep}\n")
        f.write("REGRESIÓN SIMBÓLICA CON DEAP — RESULTADOS\n")
        f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{sep}\n\n")

        f.write(f"Función objetivo : {TARGET_EXPR_STR}\n")
        f.write(f"Población        : {POPULATION_SIZE}\n")
        f.write(f"Generaciones     : {N_GENERATIONS}\n")
        f.write(f"P(cruza)         : {CROSSOVER_PROB}\n")
        f.write(f"P(mutación)      : {MUTATION_PROB}\n")
        f.write(f"Peso complejidad : {COMPLEXITY_WEIGHT}\n")
        f.write(f"Umbral parada    : {EARLY_STOP_THRESHOLD}\n")
        f.write(f"Parada anticipada: {'Sí' if early_stopped else 'No'}\n\n")

        f.write(f"{dash}\nMEJOR INDIVIDUO\n{dash}\n")
        f.write(f"Árbol       : {hof[0]}\n")
        f.write(f"Nodos       : {len(hof[0])}\n")
        f.write(f"Profundidad : {hof[0].height}\n")
        f.write(f"Fitness     : {hof[0].fitness.values[0]:.6e}\n")
        if simplified_expr is not None:
            f.write(f"Expresión   : {simplified_expr}\n")
        f.write("\n")

        if metrics:
            f.write(f"{dash}\nMÉTRICAS DE CALIDAD\n{dash}\n")
            for k, v in metrics.items():
                f.write(f"  {k:6s}: {v:.6e}\n")
            f.write("\n")

        f.write(f"{dash}\nEVOLUCIÓN POR GENERACIÓN\n{dash}\n")
        f.write(f"{'Gen':>5}  {'MSE mín':>14}  {'Tam. prom':>10}\n")
        for g, m, s in zip(gens, mins, avg_size):
            f.write(f"{g:>5}  {m:>14.6e}  {s:>10.2f}\n")

    print(f"Resultados guardados: {path}")
    return path


# ══════════════════════════════════════════════════════════════
# SECCIÓN 12 — MAIN
# ══════════════════════════════════════════════════════════════

def main():
    random.seed(RANDOM_SEED)
    output_dir = Path(__file__).resolve().parent

    print(f"\n{'═'*60}")
    print(f"  Regresión Simbólica con DEAP")
    print(f"  Función objetivo : {TARGET_EXPR_STR}")
    print(f"  Población        : {POPULATION_SIZE}  |  Generaciones: {N_GENERATIONS}")
    print(f"  Peso complejidad : {COMPLEXITY_WEIGHT}  |  Umbral parada: {EARLY_STOP_THRESHOLD}")
    print(f"{'═'*60}\n")

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats_fit   = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size  = tools.Statistics(len)
    stats_depth = tools.Statistics(lambda ind: ind.height)

    mstats = tools.MultiStatistics(
        fitness=stats_fit,
        size=stats_size,
        depth=stats_depth,
    )
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log, best_per_gen, diversity_per_gen, early_stopped = run_evolution(
        pop, toolbox,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=N_GENERATIONS,
        stats=mstats,
        halloffame=hof,
        verbose=True,
    )

    return pop, log, hof, best_per_gen, diversity_per_gen, early_stopped, output_dir


if __name__ == "__main__":
    pop, log, hof, best_per_gen, diversity_per_gen, early_stopped, output_dir = main()

    # ── Imprimir resultados ───────────────────────────────────
    print(f"\n{'═'*60}")
    print("MEJOR INDIVIDUO")
    print(f"{'═'*60}")
    print(f"Árbol       : {hof[0]}")
    print(f"Nodos       : {len(hof[0])}  |  Profundidad: {hof[0].height}")
    print(f"Fitness     : {hof[0].fitness.values[0]:.6e}")

    simplified_expr = simplify_best_individual(hof[0])
    if simplified_expr is not None:
        print(f"Expresión   : {simplified_expr}")

    # ── Métricas adicionales ──────────────────────────────────
    print(f"\n{'─'*40}")
    print("MÉTRICAS DE CALIDAD")
    print(f"{'─'*40}")
    metrics = compute_metrics(hof[0], EVAL_POINTS)
    if metrics:
        for k, v in metrics.items():
            print(f"  {k:6s}: {v:.6e}")

    # ── Guardar texto ─────────────────────────────────────────
    save_results(hof, simplified_expr, metrics, log,
                 output_dir, early_stopped, best_per_gen)

    # ── Gráficas ──────────────────────────────────────────────
    show_plots(log, hof, best_per_gen, diversity_per_gen, output_dir)
    plot_expression_tree(hof[0], output_dir)

    plt.show()
