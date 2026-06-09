
import operator
import math
import random
import datetime
import sys
from pathlib import Path
from functools import partial

import numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
import networkx as nx

try:
    from PyQt6 import QtCore, QtGui, QtWidgets
    QT_AVAILABLE = True
except Exception:
    QtCore = QtGui = QtWidgets = None
    QT_AVAILABLE = False

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


def set_target_expression(expr_str: str):
    """Actualiza la función objetivo usada por la evaluación."""
    global TARGET_EXPR_STR, _target_fn
    TARGET_EXPR_STR = expr_str
    _target_fn = build_target_function(expr_str)


def generate_random_target_expression(max_depth: int = 3, require_both: bool = False):
    """Genera una expresión simbólica aleatoria y relativamente estable.

    If require_both=True the expression is guaranteed to contain both `x` and `y`.
    """
    terminals = ["x", "y", "-3", "-2", "-1", "0", "1", "2", "3"]

    def build(depth: int):
        if depth <= 0:
            return random.choice(terminals)

        op = random.choice(["add", "sub", "mul", "div", "sin", "cos", "sqrt", "log", "abs", "neg"])

        if op == "add":
            return f"({build(depth - 1)} + {build(depth - 1)})"
        if op == "sub":
            return f"({build(depth - 1)} - {build(depth - 1)})"
        if op == "mul":
            return f"({build(depth - 1)} * {build(depth - 1)})"
        if op == "div":
            numerator = build(depth - 1)
            denominator = build(depth - 1)
            return f"({numerator} / (Abs({denominator}) + 1))"
        if op == "sin":
            return f"sin({build(depth - 1)})"
        if op == "cos":
            return f"cos({build(depth - 1)})"
        if op == "sqrt":
            return f"sqrt(Abs({build(depth - 1)}))"
        if op == "log":
            return f"log(Abs({build(depth - 1)}) + 1e-3)"
        if op == "abs":
            return f"Abs({build(depth - 1)})"
        return f"(-{build(depth - 1)})"

    # Asegurar que la expresión contenga variables según el requisito
    for _ in range(24):
        expr = build(max_depth)
        has_x = "x" in expr
        has_y = "y" in expr
        if require_both:
            if has_x and has_y:
                return expr
        else:
            if has_x or has_y:
                return expr
    return expr


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
pset.addPrimitive(math.sin,     1)
pset.addPrimitive(math.cos,     1)
pset.addPrimitive(protectedSqrt, 1)
pset.addPrimitive(protectedLog,  1)
pset.addPrimitive(protectedExp,  1)
pset.addPrimitive(protectedAbs,  1)

pset.addEphemeralConstant("rand101", partial(random.uniform, -3.0, 3.0))
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
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
            pred = func(x, y)
            # Convert to float if possible and guard against extreme values
            if isinstance(pred, (list, tuple, numpy.ndarray)):
                # We expect scalar predictions; if not, penalize
                return (PENALTY,)
            pred = float(pred)
            if not math.isfinite(pred) or abs(pred) > 1e6:
                return (PENALTY,)

            true = target_function(x, y)
            if isinstance(true, (list, tuple, numpy.ndarray)):
                return (PENALTY,)
            true = float(true)

            error = pred - true
            sq = error * error
        except (OverflowError, ZeroDivisionError, ValueError, FloatingPointError, TypeError):
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
    def _safe_for_log_plot(values):
        arr = numpy.asarray(values, dtype=float)
        arr = numpy.where(numpy.isfinite(arr), arr, 1e300)
        arr = numpy.where(arr <= 0, 1e-300, arr)
        return numpy.clip(arr, 1e-300, 1e300)

    def _ensure_2d(values, shape_like):
        arr = numpy.asarray(values, dtype=float)
        if arr.ndim == 0:
            arr = numpy.full(shape_like, float(arr))
        elif arr.shape != shape_like:
            arr = numpy.broadcast_to(arr, shape_like)
        return numpy.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)

    generations  = log.select("gen")
    min_fitness  = log.chapters["fitness"].select("min")
    avg_fitness  = log.chapters["fitness"].select("avg")
    avg_size     = log.chapters["size"].select("avg")
    max_size     = log.chapters["size"].select("max")
    avg_depth    = log.chapters["depth"].select("avg")
    max_depth    = log.chapters["depth"].select("max")

    best_fitness_history = [ind.fitness.values[0] for ind in best_per_gen]
    gens_plot = generations[: len(diversity_per_gen)]
    min_fitness_safe = _safe_for_log_plot(min_fitness)
    avg_fitness_safe = _safe_for_log_plot(avg_fitness)
    best_fitness_safe = _safe_for_log_plot(best_fitness_history)

    # ── Panel principal: 2×3 subplots ──────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Análisis Evolutivo — Regresión Simbólica DEAP",
                 fontsize=16, fontweight="bold")

    # 1. Evolución del MSE (escala log)
    ax = axes[0, 0]
    ax.semilogy(generations, min_fitness_safe, "o-", ms=3,
                color="#e74c3c", label="MSE mínimo")
    ax.semilogy(generations, avg_fitness_safe, "s-", ms=3,
                color="#3498db", alpha=0.6, label="MSE promedio")
    ax.set_title("Evolución del Error (MSE)")
    ax.set_xlabel("Generación"); ax.set_ylabel("MSE (log)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Fitness del mejor individuo
    ax = axes[0, 1]
    ax.semilogy(gens_plot, best_fitness_safe, "o-", ms=3, color="#1abc9c")
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

    Z_real_raw = target_function(X, Y)
    Z_pred_raw = numpy.array([
        [best_func(xi, yi) for xi, yi in zip(rx, ry)]
        for rx, ry in zip(X, Y)
    ], dtype=float)

    Z_real = _ensure_2d(Z_real_raw, X.shape)
    Z_pred = _ensure_2d(Z_pred_raw, X.shape)
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
    path.parent.mkdir(parents=True, exist_ok=True)

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


def run_experiment(target_expr=None, random_target=False, random_depth=3,
                   seed=RANDOM_SEED, verbose=True, require_both=False,
                   include_internal=False):
    """Ejecuta una corrida completa y devuelve todos los artefactos generados."""
    random.seed(seed)

    if random_target or not target_expr:
        target_expr = generate_random_target_expression(random_depth, require_both=require_both)

    set_target_expression(target_expr)

    output_dir = Path(__file__).resolve().parent
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
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
        verbose=verbose,
    )

    simplified_expr = simplify_best_individual(hof[0])
    metrics = compute_metrics(hof[0], EVAL_POINTS)
    results_path = save_results(
        hof, simplified_expr, metrics, log,
        output_dir, early_stopped, best_per_gen,
    )
    evo_path, surf_path = show_plots(log, hof, best_per_gen, diversity_per_gen, output_dir)
    tree_path = plot_expression_tree(hof[0], output_dir)

    result = {
        "target_expr": target_expr,
        "population_size": POPULATION_SIZE,
        "generations": N_GENERATIONS,
        "early_stopped": early_stopped,
        "simplified_expr": simplified_expr,
        "metrics": metrics,
        "results_path": results_path,
        "evo_path": evo_path,
        "surf_path": surf_path,
        "tree_path": tree_path,
        "output_dir": output_dir,
    }

    if include_internal:
        result.update({
            "pop": pop,
            "log": log,
            "hof": hof,
            "best_per_gen": best_per_gen,
            "diversity_per_gen": diversity_per_gen,
        })

    return result


def format_experiment_summary(result):
    """Devuelve un resumen legible para la GUI o la consola."""
    lines = [
        f"Función objetivo : {result['target_expr']}",
        f"Población        : {result['population_size']}",
        f"Generaciones     : {result['generations']}",
        f"Parada anticipada: {'Sí' if result['early_stopped'] else 'No'}",
        "",
        "MEJOR INDIVIDUO",
    ]

    hof = result.get("hof")
    if hof:
        lines.extend([
            f"Árbol       : {hof[0]}",
            f"Nodos       : {len(hof[0])}",
            f"Profundidad : {hof[0].height}",
            f"Fitness     : {hof[0].fitness.values[0]:.6e}",
        ])
    else:
        lines.extend([
            f"Árbol       : (no incluido en esta salida)",
            f"Fitness     : {result.get('fitness', 'n/a')}",
        ])

    if result["simplified_expr"] is not None:
        lines.append(f"Expresión   : {result['simplified_expr']}")

    if result["metrics"]:
        lines.extend(["", "MÉTRICAS DE CALIDAD"])
        for key, value in result["metrics"].items():
            lines.append(f"  {key:6s}: {value:.6e}")

    lines.extend([
        "",
        f"Resultados  : {result['results_path']}",
        f"Evolución   : {result['evo_path']}",
        f"Superficies : {result['surf_path']}",
        f"Árbol       : {result['tree_path']}",
    ])
    return "\n".join(lines)


if QT_AVAILABLE:
    class ExperimentWorker(QtCore.QObject):
        finished = QtCore.pyqtSignal(dict)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, target_expr, random_target, random_depth, require_both=False):
            super().__init__()
            self.target_expr = target_expr
            self.random_target = random_target
            self.random_depth = random_depth
            self.require_both = require_both

        @QtCore.pyqtSlot()
        def run(self):
            try:
                result = run_experiment(
                    target_expr=self.target_expr,
                    random_target=self.random_target,
                    random_depth=self.random_depth,
                    verbose=True,
                    require_both=self.require_both,
                    include_internal=False,
                )
            except Exception as exc:
                self.failed.emit(str(exc))
            else:
                self.finished.emit(result)


    class PlotImageView(QtWidgets.QScrollArea):
        """Visor de imagen con autoajuste al espacio disponible."""
        def __init__(self):
            super().__init__()
            self.setWidgetResizable(True)
            self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

            self.label = QtWidgets.QLabel("Las gráficas aparecerán aquí.")
            self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.label.setStyleSheet("background: #111827; color: #e5e7eb; border: 1px solid #374151;")
            self.label.setMinimumSize(320, 220)
            self.setWidget(self.label)

            self._source_pixmap = None

        def clear_image(self):
            self._source_pixmap = None
            self.label.setPixmap(QtGui.QPixmap())
            self.label.setText("Las gráficas aparecerán aquí.")

        def set_image(self, path):
            pixmap = QtGui.QPixmap(str(path))
            if pixmap.isNull():
                self._source_pixmap = None
                self.label.setPixmap(QtGui.QPixmap())
                self.label.setText(f"No se pudo cargar: {path}")
                return

            self._source_pixmap = pixmap
            self.label.setText("")
            self._fit_to_viewport()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._fit_to_viewport()

        def _fit_to_viewport(self):
            if self._source_pixmap is None:
                return

            vp_size = self.viewport().size()
            if vp_size.width() <= 1 or vp_size.height() <= 1:
                return

            scaled = self._source_pixmap.scaled(
                vp_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.label.setPixmap(scaled)
            self.label.resize(scaled.size())


    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Regresión Simbólica con DEAP")
            self.resize(1400, 900)

            self._thread = None
            self._worker = None
            self._build_ui()

        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)

            root_layout = QtWidgets.QHBoxLayout(central)

            controls = QtWidgets.QFrame()
            controls.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            controls.setMinimumWidth(360)
            control_layout = QtWidgets.QVBoxLayout(controls)

            title = QtWidgets.QLabel("Regresión simbólica")
            title.setStyleSheet("font-size: 24px; font-weight: 700;")
            subtitle = QtWidgets.QLabel("Ingresa una función o genera una al azar y ejecuta la búsqueda.")
            subtitle.setWordWrap(True)

            self.expr_edit = QtWidgets.QLineEdit(TARGET_EXPR_STR)
            self.expr_edit.setPlaceholderText("Ejemplo: x**2 + y**2*x**2 + 1")

            self.random_check = QtWidgets.QCheckBox("Usar función aleatoria")
            self.random_depth = QtWidgets.QSpinBox()
            self.random_depth.setRange(1, 6)
            self.random_depth.setValue(3)
            self.random_depth.setSuffix(" niveles")
            self.require_both_cb = QtWidgets.QCheckBox("Forzar x y")

            self.generate_button = QtWidgets.QPushButton("Generar función aleatoria")
            self.run_button = QtWidgets.QPushButton("Ejecutar aproximación")
            self.run_button.setMinimumHeight(44)

            self.status_label = QtWidgets.QLabel("Listo.")
            self.status_label.setWordWrap(True)

            control_layout.addWidget(title)
            control_layout.addWidget(subtitle)
            control_layout.addSpacing(12)
            control_layout.addWidget(QtWidgets.QLabel("Función objetivo"))
            control_layout.addWidget(self.expr_edit)
            control_layout.addWidget(self.random_check)
            control_layout.addWidget(self.require_both_cb)
            control_layout.addWidget(QtWidgets.QLabel("Profundidad de la función aleatoria"))
            control_layout.addWidget(self.random_depth)
            control_layout.addWidget(self.generate_button)
            control_layout.addWidget(self.run_button)
            control_layout.addStretch(1)
            control_layout.addWidget(QtWidgets.QLabel("Estado"))
            control_layout.addWidget(self.status_label)

            self.tabs = QtWidgets.QTabWidget()

            self.summary_text = QtWidgets.QPlainTextEdit()
            self.summary_text.setReadOnly(True)

            self.metrics_text = QtWidgets.QPlainTextEdit()
            self.metrics_text.setReadOnly(True)

            self.plots_tabs = QtWidgets.QTabWidget()
            self.evolution_view = PlotImageView()
            self.surface_view = PlotImageView()
            self.tree_view = PlotImageView()

            self.plots_tabs.addTab(self.evolution_view, "Evolución")
            self.plots_tabs.addTab(self.surface_view, "Superficies")
            self.plots_tabs.addTab(self.tree_view, "Árbol")

            self.tabs.addTab(self.summary_text, "Resumen")
            self.tabs.addTab(self.metrics_text, "Métricas")
            self.tabs.addTab(self.plots_tabs, "Gráficas")

            root_layout.addWidget(controls, 0)
            root_layout.addWidget(self.tabs, 1)

            self.generate_button.clicked.connect(self._generate_random_expression)
            self.run_button.clicked.connect(self._run_experiment)

        def _generate_random_expression(self):
            expr = generate_random_target_expression(self.random_depth.value(), require_both=self.require_both_cb.isChecked())
            self.expr_edit.setText(expr)
            self.random_check.setChecked(True)
            self.status_label.setText(f"Función aleatoria generada: {expr}")

        def _set_running(self, running: bool):
            self.run_button.setEnabled(not running)
            self.generate_button.setEnabled(not running)
            self.expr_edit.setEnabled(not running)
            self.random_check.setEnabled(not running)
            self.random_depth.setEnabled(not running)

        def _run_experiment(self):
            target_expr = self.expr_edit.text().strip()
            use_random = self.random_check.isChecked()
            if use_random:
                # If the user already generated a function (expr_edit non-empty), use it.
                # Only generate a new random function if the field is empty.
                if not target_expr:
                    target_expr = generate_random_target_expression(
                        self.random_depth.value(), require_both=self.require_both_cb.isChecked()
                    )
                    self.expr_edit.setText(target_expr)
            elif not target_expr:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Función requerida",
                    "Ingresa una función válida o activa 'Usar función aleatoria'.",
                )
                return

            self.summary_text.setPlainText("")
            self.metrics_text.setPlainText("")
            self.evolution_view.clear_image()
            self.surface_view.clear_image()
            self.tree_view.clear_image()

            self._set_running(True)
            self.status_label.setText("Ejecutando evolución... esto puede tardar unos minutos.")

            self._thread = QtCore.QThread(self)
            # Pass the exact expression to the worker; do not request the worker to generate another random.
            self._worker = ExperimentWorker(
                target_expr,
                False,
                self.random_depth.value(),
                require_both=self.require_both_cb.isChecked(),
            )
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._on_finished)
            self._worker.failed.connect(self._on_failed)
            self._worker.finished.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._worker.finished.connect(self._worker.deleteLater)
            self._worker.failed.connect(self._worker.deleteLater)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.start()

        def _on_finished(self, result):
            self._set_running(False)
            self.status_label.setText("Ejecución terminada.")

            summary_text = format_experiment_summary(result)
            self.summary_text.setPlainText(summary_text)

            metrics = result["metrics"] or {}
            if metrics:
                metrics_lines = [f"{key}: {value:.6e}" for key, value in metrics.items()]
            else:
                metrics_lines = ["No se pudieron calcular métricas válidas."]
            self.metrics_text.setPlainText("\n".join(metrics_lines))

            # Defer heavy pixmap loading until the event loop is idle again.
            QtCore.QTimer.singleShot(0, lambda: self.evolution_view.set_image(result["evo_path"]))
            QtCore.QTimer.singleShot(0, lambda: self.surface_view.set_image(result["surf_path"]))
            QtCore.QTimer.singleShot(0, lambda: self.tree_view.set_image(result["tree_path"]))

        def _on_failed(self, message):
            self._set_running(False)
            self.status_label.setText("La ejecución falló.")
            QtWidgets.QMessageBox.critical(self, "Error", message)


def run_cli():
    result = run_experiment(target_expr=TARGET_EXPR_STR, random_target=False, verbose=True)
    print(f"\n{'═'*60}")
    print("MEJOR INDIVIDUO")
    print(f"{'═'*60}")
    print(format_experiment_summary(result))
    return result


# ══════════════════════════════════════════════════════════════
# SECCIÓN 12 — MAIN
# ══════════════════════════════════════════════════════════════

def main():
    return run_experiment(target_expr=TARGET_EXPR_STR, random_target=False, verbose=True)


if __name__ == "__main__":
    if QT_AVAILABLE and "--cli" not in sys.argv:
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    run_cli()
