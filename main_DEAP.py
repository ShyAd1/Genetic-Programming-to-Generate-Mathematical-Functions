
import operator
import math
import random
import datetime
import sys
from pathlib import Path
from functools import partial

import numpy as np
numpy = np  # alias for legacy references
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
TARGET_EXPR_STR = "x" # Placeholder

# Parámetros del algoritmo
POPULATION_SIZE      = 300
N_GENERATIONS        = 100
CROSSOVER_PROB       = 0.80
MUTATION_PROB        = 0.20
TREE_MIN_DEPTH       = 1
TREE_MAX_DEPTH       = 4   # población inicial más compacta
TREE_HEIGHT_LIMIT    = 6   # límite duro: máximo 6 niveles de profundidad

# Penalización por complejidad — suficientemente alta para desincentivar bloat
# Un árbol de 20 nodos añade 0.02 al fitness; uno de 5 nodos añade 0.005
COMPLEXITY_WEIGHT    = 0.001

# Parada anticipada: detiene la evolución si MSE < umbral
EARLY_STOP_THRESHOLD = 1e-2

# Puntos de evaluación: cuadrícula [-100, 100] x [-100, 100]
EVAL_RANGE           = range(-100, 101)  # puntos cada 0.1 en [-10, 10]
EVAL_POINTS          = [
    (x / 10.0, y / 10.0) for x in EVAL_RANGE for y in EVAL_RANGE
]

# Arrays vectorizados para evaluación rápida (se calculan una sola vez)
# EVAL_POINTS = [(x,y) for x in EVAL_RANGE for y in EVAL_RANGE]
# → x varía en el loop externo, y en el interno
_EVAL_X = np.array([x / 10.0 for x in EVAL_RANGE for y in EVAL_RANGE])
_EVAL_Y = np.array([y / 10.0 for x in EVAL_RANGE for y in EVAL_RANGE])
_TARGET_VALUES = None  # se inicializa al cargar la función objetivo

# Semilla para reproducibilidad
RANDOM_SEED          = random.randint(0, 10000)


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
        # print(f"✔ Función objetivo cargada: f(x,y) = {sympy_expr}")
        return lambdified
    except Exception as e:
        raise ValueError(f"No se pudo parsear la función objetivo '{expr_str}': {e}")


_target_fn = build_target_function(TARGET_EXPR_STR)

def target_function(x, y):
    return _target_fn(x, y)


def set_target_expression(expr_str: str):
    """Actualiza la función objetivo usada por la evaluación."""
    global TARGET_EXPR_STR, _target_fn, _TARGET_VALUES
    TARGET_EXPR_STR = expr_str
    _target_fn = build_target_function(expr_str)
    # Precomputar valores objetivo vectorizados una sola vez
    try:
        raw = _target_fn(_EVAL_X, _EVAL_Y)
        arr = np.asarray(raw, dtype=float).ravel()
        mask = np.isfinite(arr)
        _TARGET_VALUES = (arr, mask)
    except Exception:
        _TARGET_VALUES = None


def _scalar_or_none(value):
    """Convierte un valor a float escalar si es finito; si no, devuelve None."""
    if isinstance(value, numpy.ndarray):
        if value.shape != ():
            return None
        value = value.item()

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _safe_target_value(x, y):
    try:
        return _scalar_or_none(target_function(x, y))
    except Exception:
        return None


def _safe_prediction_value(func, x, y):
    try:
        return _scalar_or_none(func(x, y))
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# SECCIÓN 3 — OPERADORES PROTEGIDOS EXTENDIDOS
# ══════════════════════════════════════════════════════════════

def protectedDiv(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(right) > 1e-10, left / right, 1.0)
    return float(result) if np.ndim(result) == 0 else result

def np_add(a, b):  return np.add(a, b)
def np_sub(a, b):  return np.subtract(a, b)
def np_mul(a, b):  return np.multiply(a, b)
def np_neg(a):     return np.negative(a)

def protectedSqrt(x):
    return np.sqrt(np.abs(x))

def protectedLog(x):
    ax = np.abs(x)
    return np.where(ax > 1e-10, np.log(ax), 0.0)

def protectedExp(x):
    with np.errstate(over="ignore"):
        return np.where(np.isfinite(x), np.exp(np.clip(x, -700, 700)), 1.0)

def protectedAbs(x):
    return np.abs(x)

def np_sin(x):
    return np.sin(x)

def np_cos(x):
    return np.cos(x)


# ══════════════════════════════════════════════════════════════
# SECCIÓN 4 — PRIMITIVOS Y CREADORES DEAP
# ══════════════════════════════════════════════════════════════

pset = gp.PrimitiveSet("MAIN", 2)

# Operadores aritméticos (wrappers NumPy — funcionan con escalares Y arrays)
pset.addPrimitive(np_add,        2)
pset.addPrimitive(np_sub,        2)
pset.addPrimitive(np_mul,        2)
pset.addPrimitive(protectedDiv,  2)
pset.addPrimitive(np_neg,        1)

# Operadores extendidos (todos compatibles con arrays NumPy)
pset.addPrimitive(np_sin,        1)
pset.addPrimitive(np_cos,        1)
pset.addPrimitive(protectedSqrt, 1)
pset.addPrimitive(protectedLog,  1)
pset.addPrimitive(protectedExp,  1)
pset.addPrimitive(protectedAbs,  1)

# Constantes enteras [-10, 10]: más compactas y útiles que floats aleatorios continuos
pset.addEphemeralConstant("rand101", partial(random.randint, -10, 10))
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

    Evaluación vectorizada con NumPy: compila el árbol GP como función
    y la aplica sobre los arrays de puntos de una sola vez, eliminando
    el costoso loop Python punto a punto.
    """
    func = toolbox.compile(expr=individual)
    PENALTY = 1e20

    # Evaluación vectorizada
    if _TARGET_VALUES is not None:
        y_true_arr, mask = _TARGET_VALUES
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                raw_pred = func(_EVAL_X, _EVAL_Y)
            y_pred_arr = np.asarray(raw_pred, dtype=float).ravel()
            if y_pred_arr.shape != y_true_arr.shape:
                y_pred_arr = np.broadcast_to(y_pred_arr, y_true_arr.shape).copy()
            valid = mask & np.isfinite(y_pred_arr)
            if not np.any(valid):
                return (PENALTY,)
            errors = y_pred_arr[valid] - y_true_arr[valid]
            mse = float(np.mean(errors ** 2))
            if not math.isfinite(mse):
                return (PENALTY,)
        except Exception:
            return (PENALTY,)
    else:
        # Fallback escalar si _TARGET_VALUES no está disponible
        sqerrors = []
        for x, y in points:
            true = _safe_target_value(x, y)
            if true is None:
                continue
            pred = _safe_prediction_value(func, x, y)
            if pred is None or abs(pred) > 1e6:
                return (PENALTY,)
            sqerrors.append((pred - true) ** 2)
        if not sqerrors:
            return (PENALTY,)
        mse = math.fsum(sqerrors) / len(sqerrors)

    complexity_bonus = COMPLEXITY_WEIGHT * len(individual)
    return (mse + complexity_bonus,)


toolbox.register("evaluate", evalSymbReg, points=EVAL_POINTS)
toolbox.register("select",   tools.selTournament, tournsize=5)
toolbox.register("mate",     gp.cxOnePointLeafBiased, termpb=0.3)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=3)
toolbox.register("mutate",       gp.mutUniform,          expr=toolbox.expr_mut, pset=pset)
toolbox.register("mutate_shrink", gp.mutShrink)
toolbox.register("mutate_node",   gp.mutNodeReplacement,  pset=pset)

toolbox.decorate("mate",        gp.staticLimit(key=operator.attrgetter("height"),
                                               max_value=TREE_HEIGHT_LIMIT))
toolbox.decorate("mutate",      gp.staticLimit(key=operator.attrgetter("height"),
                                               max_value=TREE_HEIGHT_LIMIT))
toolbox.decorate("mutate_node", gp.staticLimit(key=operator.attrgetter("height"),
                                               max_value=TREE_HEIGHT_LIMIT))


# ══════════════════════════════════════════════════════════════
# SECCIÓN 6 — BUCLE EVOLUTIVO CON PARADA ANTICIPADA
# ══════════════════════════════════════════════════════════════

def run_evolution(pop, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose=True):
    """
    Bucle evolutivo personalizado con:
      - Elitismo (preserva el mejor individuo)
      - Mutación mixta: uniforme (50%), shrink (30%), reemplazo de nodo (20%)
      - Poda activa: cualquier árbol que supere TREE_HEIGHT_LIMIT se reemplaza
      - Parada anticipada cuando MSE < EARLY_STOP_THRESHOLD
    """
    logbook         = tools.Logbook()
    logbook.header  = ["gen", "nevals"] + (stats.fields if stats else [])
    best_per_gen    = []
    diversity_per_gen = []

    def _apply_mutation(ind):
        """Aplica una de tres mutaciones según probabilidad."""
        r = random.random()
        if r < 0.50:
            toolbox.mutate(ind)       # mutUniform: reemplaza subárbol
        elif r < 0.80:
            toolbox.mutate_shrink(ind) # shrink: contrae subárbol a un terminal
        else:
            toolbox.mutate_node(ind)  # nodeReplacement: cambia un nodo
        del ind.fitness.values
        return ind

    def _enforce_height(ind):
        """Si el árbol supera el límite, aplica shrink hasta que cumpla."""
        attempts = 0
        while ind.height > TREE_HEIGHT_LIMIT and attempts < 5:
            toolbox.mutate_shrink(ind)
            del ind.fitness.values
            attempts += 1
        return ind

    # ── Generación 0 ────────────────────────────────────────
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

    # ── Generaciones 1..ngen ────────────────────────────────
    for gen in range(1, ngen + 1):
        elite = toolbox.clone(halloffame[0])
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop) - 1)))

        # Cruza
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        # Mutación mixta
        for mutant in offspring:
            if random.random() < mutpb:
                _apply_mutation(mutant)

        # Poda activa de árboles fuera de límite
        for ind in offspring:
            if ind.height > TREE_HEIGHT_LIMIT:
                _enforce_height(ind)

        # Evaluar modificados
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring + [elite]

        if halloffame is not None:
            halloffame.update(pop)

        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

        best_per_gen.append(toolbox.clone(halloffame[0]))
        diversity_per_gen.append(len(set(str(i) for i in pop)) / len(pop))

        current_best = halloffame[0].fitness.values[0]
        if current_best < EARLY_STOP_THRESHOLD:
            early_stopped = True
            break

    return pop, logbook, best_per_gen, diversity_per_gen, early_stopped


# ══════════════════════════════════════════════════════════════
# SECCIÓN 7 — SIMPLIFICACIÓN SIMBÓLICA (SymPy)
# ══════════════════════════════════════════════════════════════

_OP_MAP = {
    "np_add":       lambda a, b: a + b,
    "np_sub":       lambda a, b: a - b,
    "np_mul":       lambda a, b: a * b,
    "protectedDiv": lambda a, b: a / b,
    "np_neg":       lambda a:    -a,
    # keep old names as fallback in case old pickled individuals exist
    "add":          lambda a, b: a + b,
    "sub":          lambda a, b: a - b,
    "mul":          lambda a, b: a * b,
    "neg":          lambda a:    -a,
    "np_sin":       lambda a:    sp.sin(a),
    "np_cos":       lambda a:    sp.cos(a),
    "protectedSqrt": lambda a:  sp.sqrt(sp.Abs(a)),
    "protectedLog":  lambda a:  sp.log(sp.Abs(a) + sp.Float(1e-10)),
    "protectedExp":  lambda a:  sp.exp(a),
    "protectedAbs":  lambda a:  sp.Abs(a),
}

def simplify_best_individual(individual):
    """Convierte el árbol DEAP a una expresión SymPy equivalente.

    La conversión evita una simplificación agresiva para que la expresión
    impresa en el reporte se mantenga alineada con el árbol original.
    """
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
                return sp.Float(node.value), idx + 1
            except Exception:
                try:
                    return sp.Float(str(node)), idx + 1
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
        return expr
    except Exception as e:
        print(f"Error al simplificar: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# SECCIÓN 8 — MÉTRICAS ADICIONALES
# ══════════════════════════════════════════════════════════════

def compute_metrics(individual, points):
    """Calcula MSE, RMSE, MAE y R² para el mejor individuo (vectorizado)."""
    func = toolbox.compile(expr=individual)

    if _TARGET_VALUES is not None:
        y_true_arr, mask = _TARGET_VALUES
        try:
            raw_pred = func(_EVAL_X, _EVAL_Y)
            y_pred_arr = np.asarray(raw_pred, dtype=float).ravel()
            if y_pred_arr.shape != y_true_arr.shape:
                y_pred_arr = np.broadcast_to(y_pred_arr, y_true_arr.shape).copy()
            valid = mask & np.isfinite(y_pred_arr)
            if not np.any(valid):
                return None
            yt = y_true_arr[valid]
            yp = y_pred_arr[valid]
        except Exception:
            return None
    else:
        yt_list, yp_list = [], []
        for x, y in points:
            true = _safe_target_value(x, y)
            pred = _safe_prediction_value(func, x, y)
            if true is None or pred is None:
                continue
            yt_list.append(true); yp_list.append(pred)
        if not yt_list:
            return None
        yt = np.array(yt_list)
        yp = np.array(yp_list)

    sq_errors = (yp - yt) ** 2
    mse    = float(np.mean(sq_errors))
    mae    = float(np.mean(np.abs(yp - yt)))
    ss_res = float(np.sum(sq_errors))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"MSE": mse, "RMSE": math.sqrt(mse), "MAE": mae, "R²": r2}


# ══════════════════════════════════════════════════════════════
# SECCIÓN 9 — VISUALIZACIÓN DEL ÁRBOL DE EXPRESIÓN
# ══════════════════════════════════════════════════════════════

def plot_expression_tree(individual, output_dir, filename="arbol_expresion.png"):
    """Dibuja el árbol de expresión GP usando NetworkX."""
    nodes, edges, labels = gp.graph(individual)

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    root = nodes[0] if nodes else 0

    def _subtree_leaf_count(node, cache):
        if node in cache:
            return cache[node]
        children = list(g.successors(node))
        if not children:
            cache[node] = 1
            return 1
        count = sum(_subtree_leaf_count(child, cache) for child in children)
        cache[node] = count
        return count

    def _tree_layout(graph, root_node, x_gap=3.1, y_gap=2.2):
        leaf_cache = {}
        _subtree_leaf_count(root_node, leaf_cache)
        positions = {}

        def _assign(node, left_x, depth):
            children = list(graph.successors(node))
            if not children:
                positions[node] = (left_x, -depth * y_gap)
                return left_x + x_gap

            current_x = left_x
            child_centers = []
            for child in children:
                current_x = _assign(child, current_x, depth + 1)
                child_centers.append(positions[child][0])

            positions[node] = (sum(child_centers) / len(child_centers), -depth * y_gap)
            return current_x

        _assign(root_node, 0.0, 0)
        return positions

    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        pos = _tree_layout(g, root)

    def _wrap_label(text, width=12):
        text = str(text)
        if len(text) <= width:
            return text
        return "\n".join(text[i:i + width] for i in range(0, len(text), width))

    max_label_len = max((len(str(label)) for label in labels.values()), default=1)
    wrap_width = 8 if max_label_len > 10 else 12
    wrapped_labels = {node: _wrap_label(labels.get(node, ""), width=wrap_width) for node in g.nodes()}

    def _estimate_node_size(text):
        lines = str(text).split("\n")
        longest = max((len(line) for line in lines), default=1)
        return max(2400, 380 * longest * len(lines))

    node_sizes = [_estimate_node_size(wrapped_labels.get(node, "")) for node in g.nodes()]
    font_size = 10 if max_label_len <= 12 else max(7, 13 - max_label_len // 5)

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

    x_values = [x for x, _ in pos.values()] if pos else [0]
    y_values = [y for _, y in pos.values()] if pos else [0]
    width = max(14, (max(x_values) - min(x_values) + 4) if x_values else 14)
    height = max(8, abs(min(y_values)) + 3 if y_values else 8)
    fig, ax = plt.subplots(figsize=(width, height))
    nx.draw(
        g, pos, ax=ax, labels=wrapped_labels, with_labels=True,
        node_color=node_colors, node_size=node_sizes, node_shape="s",
        font_size=font_size, font_color="white", font_weight="bold",
        arrows=True, arrowsize=12, edge_color="#7f8c8d", width=1.5,
    )
    ax.margins(x=0.12, y=0.18)
    if pos:
        ax.set_xlim(min(x_values) - 2.0, max(x_values) + 2.0)
        ax.set_ylim(min(y_values) - 2.5, max(y_values) + 2.5)
    ax.set_axis_off()

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
        arr = numpy.where(numpy.isfinite(arr), arr, numpy.nan)
        return arr

    def _evaluate_surface(func, X, Y):
        """Evalúa func sobre la cuadrícula X,Y de forma vectorizada."""
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                Z = np.asarray(func(X, Y), dtype=float)
            if Z.shape != X.shape:
                Z = np.broadcast_to(Z, X.shape).copy()
            Z[~np.isfinite(Z)] = np.nan
            return Z
        except Exception:
            # fallback escalar
            surface = numpy.empty(X.shape, dtype=float)
            for r in range(X.shape[0]):
                for c in range(X.shape[1]):
                    v = _scalar_or_none(func(X[r, c], Y[r, c]))
                    surface[r, c] = numpy.nan if v is None else v
            return surface

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
    x = numpy.linspace(-5, 5, 40)
    y = numpy.linspace(-5, 5, 40)
    X, Y = numpy.meshgrid(x, y)

    Z_real_raw = _evaluate_surface(target_function, X, Y)
    Z_pred_raw = _evaluate_surface(best_func, X, Y)

    Z_real = _ensure_2d(Z_real_raw, X.shape)
    Z_pred = _ensure_2d(Z_pred_raw, X.shape)
    Z_err = numpy.abs(Z_real - Z_pred)

    fig = plt.figure(figsize=(18, 5))
    titles   = ["Función objetivo", "Mejor aproximación DEAP", "Error absoluto |f − f̂|"]
    Z_list   = [Z_real, Z_pred, Z_err]
    cmaps    = ["viridis", "plasma", "hot"]

    for k, (title, Z, cmap) in enumerate(zip(titles, Z_list, cmaps)):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.plot_surface(X, Y, numpy.ma.masked_invalid(Z), cmap=cmap, edgecolor="none")
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


def run_experiment(target_expr=None, seed=RANDOM_SEED, verbose=True,
                   include_internal=False):
    """Ejecuta una corrida completa y devuelve todos los artefactos generados."""
    random.seed(seed)
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
        "log": log,  # always include for the GUI evolution table
    }

    if include_internal:
        result.update({
            "pop": pop,
            "hof": hof,
            "best_per_gen": best_per_gen,
            "diversity_per_gen": diversity_per_gen,
        })

    return result


if QT_AVAILABLE:
    class ExperimentWorker(QtCore.QObject):
        finished = QtCore.pyqtSignal(dict)
        failed = QtCore.pyqtSignal(str)

        def __init__(self, target_expr):
            super().__init__()
            self.target_expr = target_expr

        @QtCore.pyqtSlot()
        def run(self):
            try:
                result = run_experiment(
                    target_expr=self.target_expr,
                    verbose=True,
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
            subtitle = QtWidgets.QLabel("Ingresa una función y ejecuta la aproximación.")
            subtitle.setWordWrap(True)

            self.expr_edit = QtWidgets.QLineEdit(TARGET_EXPR_STR)
            self.expr_edit.setPlaceholderText("Ejemplo: x**2 + y**2*x**2 + 1")

            self.run_button = QtWidgets.QPushButton("Ejecutar aproximación")
            self.run_button.setMinimumHeight(44)

            self.status_label = QtWidgets.QLabel("Listo.")
            self.status_label.setWordWrap(True)

            control_layout.addWidget(title)
            control_layout.addWidget(subtitle)
            control_layout.addSpacing(12)
            control_layout.addWidget(QtWidgets.QLabel("Función objetivo"))
            control_layout.addWidget(self.expr_edit)
            control_layout.addWidget(self.run_button)
            control_layout.addStretch(1)
            control_layout.addWidget(QtWidgets.QLabel("Estado"))
            control_layout.addWidget(self.status_label)

            self.tabs = QtWidgets.QTabWidget()

            # ── Sub-pestañas de Resumen ───────────────────────────
            self.summary_tabs = QtWidgets.QTabWidget()

            self.summary_text = QtWidgets.QPlainTextEdit()
            self.summary_text.setReadOnly(True)
            self.summary_text.setFont(QtGui.QFont("Courier New", 9))

            self.evolution_table = QtWidgets.QTableWidget()
            self.evolution_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.evolution_table.setAlternatingRowColors(True)
            self.evolution_table.horizontalHeader().setStretchLastSection(True)
            self.evolution_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.evolution_table.setSortingEnabled(True)

            evo_table_desc = QtWidgets.QLabel(
                "<b>Estadísticas por generación del algoritmo genético</b><br>"
                "<b>Gen:</b> número de generación (0 = población inicial sin evolucionar).  "
                "<b>Nevals:</b> individuos re-evaluados (solo los modificados por cruza/mutación).<br>"
                "<b>Depth avg/min/max/std:</b> profundidad de los árboles. Creciente = posible bloat; muy baja = expresiones simples sin convergencia.<br>"
                "<b>Fitness avg/min/max/std:</b> distribución del fitness (MSE + penalización por complejidad). El mínimo es el mejor individuo de esa generación.<br>"
                "<b>Size avg/min/max/std:</b> nodos por árbol. COMPLEXITY_WEIGHT controla su crecimiento para evitar inflación de código.<br>"
                "Haz clic en cualquier encabezado de columna para ordenar."
            )
            evo_table_desc.setWordWrap(True)
            evo_table_desc.setTextFormat(QtCore.Qt.TextFormat.RichText)
            evo_table_desc.setStyleSheet("padding: 8px; background: #1e293b; border-radius: 4px; color: #cbd5e1;")

            evo_table_widget = QtWidgets.QWidget()
            evo_table_layout = QtWidgets.QVBoxLayout(evo_table_widget)
            evo_table_layout.setContentsMargins(0, 0, 0, 0)
            evo_table_layout.addWidget(evo_table_desc)
            evo_table_layout.addWidget(self.evolution_table, 1)

            self.summary_tabs.addTab(self.summary_text, "Reporte")
            self.summary_tabs.addTab(evo_table_widget, "Evolución")

            self.metrics_text = QtWidgets.QPlainTextEdit()
            self.metrics_text.setReadOnly(True)

            metrics_desc = QtWidgets.QLabel(
                "<b>Métricas de calidad del mejor individuo encontrado</b><br>"
                "<b>MSE</b> (Error Cuadrático Medio): promedio de los errores al cuadrado entre la función objetivo y la aproximación. "
                "Penaliza fuertemente errores grandes. Cuanto más cercano a 0, mejor.<br>"
                "<b>RMSE</b> (Raíz del MSE): misma unidad que la función. Más interpretable que el MSE directamente.<br>"
                "<b>MAE</b> (Error Absoluto Medio): promedio de los errores absolutos. Menos sensible a outliers que el MSE.<br>"
                "<b>R²</b> (Coeficiente de determinación): fracción de la varianza de la función objetivo explicada por la aproximación. "
                "R²=1.0 es ajuste perfecto; R²=0.0 equivale a predecir siempre la media; valores negativos indican un ajuste peor que la media."
            )
            metrics_desc.setWordWrap(True)
            metrics_desc.setTextFormat(QtCore.Qt.TextFormat.RichText)
            metrics_desc.setStyleSheet("padding: 8px; background: #1e293b; border-radius: 4px; color: #cbd5e1;")

            metrics_widget = QtWidgets.QWidget()
            metrics_layout = QtWidgets.QVBoxLayout(metrics_widget)
            metrics_layout.setContentsMargins(0, 0, 0, 0)
            metrics_layout.addWidget(metrics_desc)
            metrics_layout.addWidget(self.metrics_text)

            self.plots_tabs = QtWidgets.QTabWidget()

            def _plot_tab(view_widget, title_html):
                w = QtWidgets.QWidget()
                vl = QtWidgets.QVBoxLayout(w)
                vl.setContentsMargins(0, 0, 0, 0)
                desc = QtWidgets.QLabel(title_html)
                desc.setWordWrap(True)
                desc.setTextFormat(QtCore.Qt.TextFormat.RichText)
                desc.setStyleSheet("padding: 8px; background: #1e293b; border-radius: 4px; color: #cbd5e1;")
                vl.addWidget(desc)
                vl.addWidget(view_widget, 1)
                return w

            self.evolution_view = PlotImageView()
            self.surface_view   = PlotImageView()
            self.tree_view      = PlotImageView()

            evo_desc = (
                "<b>Gráfica de evolución del algoritmo genético (6 paneles)</b><br>"
                "<b>① MSE mínimo y promedio (log):</b> muestra cómo cae el error en la población a lo largo de las generaciones. "
                "Una caída rápida indica buena convergencia; si el mínimo baja pero el promedio no, la población ha perdido diversidad.<br>"
                "<b>② Fitness del mejor individuo (log):</b> trayectoria del campeón del Hall of Fame. "
                "La línea roja punteada marca el umbral de parada anticipada.<br>"
                "<b>③ Diversidad genética:</b> fracción de árboles únicos en la población. "
                "Valores bajos (&lt;0.2) indican convergencia prematura o pérdida de diversidad.<br>"
                "<b>④ Complejidad — nodos por árbol:</b> si el tamaño promedio crece sin control se produce 'bloat' (inflación de código). "
                "El parámetro COMPLEXITY_WEIGHT penaliza esto.<br>"
                "<b>⑤ Profundidad de los árboles:</b> una profundidad creciente aumenta el espacio de búsqueda pero también el tiempo de evaluación.<br>"
                "<b>⑥ Tamaño del mejor individuo:</b> si el campeón crece mucho sin mejorar el error, la solución es probablemente redundante."
            )
            surf_desc = (
                "<b>Comparación de superficies 3D en la cuadrícula [-5, 5] × [-5, 5]</b><br>"
                "<b>Izquierda — Función objetivo:</b> la superficie real f(x,y) que el algoritmo intenta aproximar.<br>"
                "<b>Centro — Mejor aproximación DEAP:</b> la expresión simbólica encontrada por el algoritmo genético. "
                "Idealmente debe tener la misma forma que la función objetivo.<br>"
                "<b>Derecha — Error absoluto |f − f̂|:</b> diferencia punto a punto entre ambas superficies. "
                "Zonas rojas/claras indican regiones donde la aproximación es menos precisa. "
                "Un error uniformemente bajo y plano es señal de un buen ajuste global."
            )
            tree_desc = (
                "<b>Árbol de expresión del mejor individuo (representación interna del árbol GP)</b><br>"
                "Cada nodo cuadrado representa una operación o terminal. "
                "<span style='color:#5dade2;'>■ Azul</span> = operador matemático (add, mul, sin, …)  "
                "<span style='color:#2ecc71;'>■ Verde claro</span> = variable (x, y)  "
                "<span style='color:#27ae60;'>■ Verde oscuro</span> = constante numérica.<br>"
                "La raíz del árbol es el nodo superior; los hijos son los argumentos de cada operador. "
                "Árboles más profundos representan expresiones más complejas. "
                "Si el árbol es muy grande comparado con su R², probablemente contiene subárboles redundantes "
                "(p.ej. x - x, o * 1.0) que SymPy eliminaría al simplificar."
            )

            self.plots_tabs.addTab(_plot_tab(self.evolution_view, evo_desc),  "Evolución")
            self.plots_tabs.addTab(_plot_tab(self.surface_view,   surf_desc), "Superficies")
            self.plots_tabs.addTab(_plot_tab(self.tree_view,      tree_desc), "Árbol")

            self.tabs.addTab(self.summary_tabs, "Resumen")
            self.tabs.addTab(metrics_widget, "Métricas")
            self.tabs.addTab(self.plots_tabs, "Gráficas")

            # ── Pestaña de Ayuda ─────────────────────────────────
            help_scroll = QtWidgets.QScrollArea()
            help_scroll.setWidgetResizable(True)
            help_content = QtWidgets.QWidget()
            help_layout = QtWidgets.QVBoxLayout(help_content)
            help_layout.setContentsMargins(20, 20, 20, 20)
            help_layout.setSpacing(12)

            help_text = QtWidgets.QLabel()
            help_text.setWordWrap(True)
            help_text.setTextFormat(QtCore.Qt.TextFormat.RichText)
            help_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
            help_text.setText("""
<h2 style="color:#60a5fa;">Regresión Simbólica con DEAP — Guía de uso</h2>

<h3 style="color:#34d399;">¿Qué hace esta aplicación?</h3>
<p>Utiliza <b>programación genética</b> para encontrar automáticamente una expresión matemática
que aproxime la función objetivo que tú defines. El algoritmo evoluciona una población de
árboles de expresión durante múltiples generaciones hasta hallar la mejor aproximación posible.</p>

<hr/>
<h3 style="color:#34d399;">Cómo usar la interfaz</h3>
<ol>
  <li><b>Escribe una función objetivo</b> en el campo de texto (por ejemplo: <code>x**2 + y</code>).</li>
  <li>Haz clic en <b>"Ejecutar aproximación"</b> para iniciar la evolución.</li>
  <li>Espera a que termine (puede tardar varios minutos según tu hardware).</li>
  <li>Consulta los resultados en las pestañas: <b>Resumen, Métricas y Gráficas</b>.</li>
</ol>

<hr/>
<h3 style="color:#34d399;">Cómo escribir funciones matemáticas</h3>
<p>Las funciones usan las variables <b>x</b> e <b>y</b>. La sintaxis es Python/SymPy:</p>

<table border="0" cellspacing="6" cellpadding="4">
  <tr><th align="left" style="color:#fbbf24;">Operación</th><th align="left" style="color:#fbbf24;">Sintaxis</th><th align="left" style="color:#fbbf24;">Ejemplo</th></tr>
  <tr><td>Suma</td><td><code>+</code></td><td><code>x + y</code></td></tr>
  <tr><td>Resta</td><td><code>-</code></td><td><code>x - y</code></td></tr>
  <tr><td>Multiplicación</td><td><code>*</code></td><td><code>x * y</code></td></tr>
  <tr><td>División</td><td><code>/</code></td><td><code>x / y</code></td></tr>
  <tr><td>Potencia</td><td><code>**</code></td><td><code>x**2</code>, <code>x**0.5</code></td></tr>
  <tr><td>Seno</td><td><code>sin(...)</code></td><td><code>sin(x)</code></td></tr>
  <tr><td>Coseno</td><td><code>cos(...)</code></td><td><code>cos(y)</code></td></tr>
  <tr><td>Raíz cuadrada</td><td><code>sqrt(...)</code></td><td><code>sqrt(x**2 + y**2)</code></td></tr>
  <tr><td>Logaritmo natural</td><td><code>log(...)</code></td><td><code>log(Abs(x) + 1)</code></td></tr>
  <tr><td>Exponencial</td><td><code>exp(...)</code></td><td><code>exp(x)</code></td></tr>
  <tr><td>Valor absoluto</td><td><code>Abs(...)</code></td><td><code>Abs(x - y)</code></td></tr>
  <tr><td>Constante pi</td><td><code>pi</code></td><td><code>sin(pi * x)</code></td></tr>
</table>

<h3 style="color:#f87171;">Consejos y advertencias</h3>
<ul>
  <li>Evita divisiones sin protección: usa <code>Abs(y) + 1</code> en el denominador.</li>
  <li>Evita <code>log</code> o <code>sqrt</code> de valores negativos; usa <code>Abs(...)</code> dentro.</li>
  <li>Las funciones muy complejas o con valores extremos pueden resultar en errores de evaluación.</li>
  <li>La función se evalúa en la cuadrícula <b>[-10, 10] × [-10, 10]</b> con paso 0.1.</li>
</ul>

<h3 style="color:#34d399;">Pestañas de resultados</h3>
<ul>
  <li><b>Resumen:</b> contenido completo del archivo TXT generado con todos los parámetros, el mejor individuo encontrado, métricas de calidad y la evolución generación por generación.</li>
  <li><b>Métricas:</b> MSE, RMSE, MAE y R² del mejor individuo encontrado.</li>
  <li><b>Gráficas → Evolución:</b> curva del fitness mínimo y tamaño de árbol por generación.</li>
  <li><b>Gráficas → Superficies:</b> comparación 3D de la función objetivo vs. la aproximada.</li>
  <li><b>Gráficas → Árbol:</b> visualización del árbol de expresión del mejor individuo.</li>
</ul>

<h3 style="color:#34d399;">Archivos generados</h3>
<p>Todos los archivos se guardan en la misma carpeta que el script:</p>
<ul>
  <li><code>resultados_YYYYMMDD_HHMMSS.txt</code> — reporte completo de texto</li>
  <li><code>evolucion_estadisticas.png</code> — gráfica de evolución</li>
  <li><code>superficies_deap.png</code> — gráfica de superficies</li>
  <li><code>arbol_expresion.png</code> — árbol de la expresión encontrada</li>
</ul>
""")
            help_layout.addWidget(help_text)
            help_layout.addStretch(1)
            help_scroll.setWidget(help_content)
            self.tabs.addTab(help_scroll, "Ayuda")

            root_layout.addWidget(controls, 0)
            root_layout.addWidget(self.tabs, 1)

            self.run_button.clicked.connect(self._run_experiment)

        def _set_running(self, running: bool):
            self.run_button.setEnabled(not running)
            self.expr_edit.setEnabled(not running)

        def _run_experiment(self):
            target_expr = self.expr_edit.text().strip()
            if not target_expr:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Función requerida",
                    "Ingresa una función válida o usa 'Generar función aleatoria'.",
                )
                return

            self.summary_text.setPlainText("")
            self.metrics_text.setPlainText("")
            self.evolution_table.clearContents()
            self.evolution_table.setRowCount(0)
            self.evolution_view.clear_image()
            self.surface_view.clear_image()
            self.tree_view.clear_image()

            self._set_running(True)
            self.status_label.setText("Ejecutando evolución... esto puede tardar unos minutos.")

            self._thread = QtCore.QThread(self)
            self._worker = ExperimentWorker(target_expr)
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

            # ── Pestaña Reporte: contenido completo del TXT ──────
            results_path = result.get("results_path")
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    summary_text = f.read()
            except Exception as e:
                summary_text = f"No se pudo leer el archivo de resultados:\n{results_path}\n\n{e}"
            self.summary_text.setPlainText(summary_text)

            # ── Pestaña Evolución: tabla de generaciones ──────────
            log = result.get("log")
            if log is not None:
                try:
                    chapters = log.chapters
                    gens    = log.select("gen")
                    nevals  = log.select("nevals")
                    # depth chapter
                    d_avg = chapters["depth"].select("avg")
                    d_min = chapters["depth"].select("min")
                    d_max = chapters["depth"].select("max")
                    d_std = chapters["depth"].select("std")
                    # fitness chapter
                    f_avg = chapters["fitness"].select("avg")
                    f_min = chapters["fitness"].select("min")
                    f_max = chapters["fitness"].select("max")
                    f_std = chapters["fitness"].select("std")
                    # size chapter
                    s_avg = chapters["size"].select("avg")
                    s_min = chapters["size"].select("min")
                    s_max = chapters["size"].select("max")
                    s_std = chapters["size"].select("std")

                    columns = [
                        "Gen", "Nevals",
                        "Depth avg", "Depth min", "Depth max", "Depth std",
                        "Fitness avg", "Fitness min", "Fitness max", "Fitness std",
                        "Size avg", "Size min", "Size max", "Size std",
                    ]
                    self.evolution_table.setSortingEnabled(False)
                    self.evolution_table.setColumnCount(len(columns))
                    self.evolution_table.setHorizontalHeaderLabels(columns)
                    self.evolution_table.setRowCount(len(gens))

                    def _item(val, is_sci=False):
                        try:
                            fval = float(val)
                            text = f"{fval:.4e}" if is_sci else f"{fval:.4f}"
                        except Exception:
                            text = str(val)
                        item = QtWidgets.QTableWidgetItem(text)
                        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                        return item

                    for row, g in enumerate(gens):
                        self.evolution_table.setItem(row, 0,  _item(g))
                        self.evolution_table.setItem(row, 1,  _item(nevals[row]))
                        self.evolution_table.setItem(row, 2,  _item(d_avg[row]))
                        self.evolution_table.setItem(row, 3,  _item(d_min[row]))
                        self.evolution_table.setItem(row, 4,  _item(d_max[row]))
                        self.evolution_table.setItem(row, 5,  _item(d_std[row]))
                        self.evolution_table.setItem(row, 6,  _item(f_avg[row], True))
                        self.evolution_table.setItem(row, 7,  _item(f_min[row], True))
                        self.evolution_table.setItem(row, 8,  _item(f_max[row], True))
                        self.evolution_table.setItem(row, 9,  _item(f_std[row], True))
                        self.evolution_table.setItem(row, 10, _item(s_avg[row]))
                        self.evolution_table.setItem(row, 11, _item(s_min[row]))
                        self.evolution_table.setItem(row, 12, _item(s_max[row]))
                        self.evolution_table.setItem(row, 13, _item(s_std[row]))

                    self.evolution_table.resizeColumnsToContents()
                    self.evolution_table.setSortingEnabled(True)
                except Exception as exc:
                    self.evolution_table.setRowCount(1)
                    self.evolution_table.setColumnCount(1)
                    self.evolution_table.setHorizontalHeaderLabels(["Error"])
                    self.evolution_table.setItem(0, 0, QtWidgets.QTableWidgetItem(str(exc)))

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


if __name__ == "__main__":
    if QT_AVAILABLE and "--cli" not in sys.argv:
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    run_experiment(target_expr=TARGET_EXPR_STR, verbose=True)
