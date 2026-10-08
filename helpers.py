import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from itertools import combinations, chain
from collections import deque
import networkx as nx


# ================================================================================
# Часть 1. Максимальный поток и минимальный разрез
# ================================================================================


def all_cuts(data):
    n = len(data[0])
    nodes = [i for i in range(n - 1)]
    combos = [list(combinations(nodes, k)) for k in range(1, n)]
    combos = list(chain(*combos))
    combos = [comb for comb in combos if 0 in comb]

    result = []
    for the_slice in combos:
        edges = []
        restricted = []
        for node in the_slice:
            for j in range(n):
                if data[node, j] != 0:
                    edges.append((node, j))
        for node_1 in the_slice:
            for node_2 in the_slice:
                restricted.append((node_1, node_2))
        edges_set = set(edges)
        restricted_set = set(restricted)
        allowed = edges_set - restricted_set

        cut_value = 0
        for edge in allowed:
            i, j = edge
            cut_value += data[i, j]
        result.append((the_slice, cut_value, allowed))

    result = sorted(result, key=lambda tup: tup[1])
    return result


def find_path(residual, s, t):
    n = residual.shape[0]
    parent = [-1] * n
    visited = [False] * n
    visited[s] = True

    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u, v] > 0:
                visited[v] = True
                parent[v] = u
                if v == t:
                    path = []
                    cur = t
                    while cur != -1:
                        path.append(cur)
                        cur = parent[cur]
                    return path[::-1]
                queue.append(v)
    return None


def max_flow(names, capacity, is_print=False):
    n = capacity.shape[0]
    residual = capacity.copy()
    value = 0
    log = []

    while True:
        path = find_path(residual, 0, n - 1)

        if path is None:
            break

        edges = [(path[k], path[k + 1]) for k in range(len(path) - 1)]
        delta = min(residual[i, j] for (i, j) in edges)

        for (i, j) in edges:
            residual[i, j] -= delta
            residual[j, i] += delta

        value += delta
        log.append((path, delta, value, np.maximum(capacity - residual, 0)))

        if is_print:
            print("Найден путь:", " -> ".join(names[v] for v in path),
                  "| delta =", delta, "| величина потока =", value)

    flow = np.maximum(capacity - residual, 0)

    return flow, log, residual


def build_graph(names, capacity):
    G = nx.DiGraph()
    G.add_nodes_from(names)
    for i in range(len(names)):
        for j in range(len(names)):
            if capacity[i, j] > 0:
                G.add_edge(names[i], names[j])

    dist = nx.shortest_path_length(G, names[0])
    last_layer = max([dist[name] for name in names if name != names[-1]]) + 1
    for name in G.nodes():
        if name == names[-1]:
            G.nodes[name]["layer"] = last_layer
        else:
            G.nodes[name]["layer"] = dist[name]
    pos = nx.multipartite_layout(G, subset_key="layer")

    return G, pos


def draw_step(G, pos, names, capacity, flow, path, title):
    path_edges = []
    if path is not None:
        for k in range(len(path) - 1):
            path_edges.append((names[path[k]], names[path[k + 1]]))

    edge_colors = []
    edge_widths = []
    edge_labels = {}
    for (u, v) in G.edges():
        i = names.index(u)
        j = names.index(v)
        edge_labels[(u, v)] = str(flow[i, j]) + "/" + str(capacity[i, j])
        if (u, v) in path_edges:
            edge_colors.append("red")
            edge_widths.append(3.5)
        elif flow[i, j] == capacity[i, j]:
            edge_colors.append("tab:gray")
            edge_widths.append(3.5)
        else:
            edge_colors.append("gray")
            edge_widths.append(1.5)

    plt.figure(figsize=(8, 4.5))
    nx.draw_networkx_nodes(G, pos, node_color="white",
                           edgecolors="black", node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           arrowsize=20, node_size=900)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, label_pos=0.35, font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_cut(G, pos, names, capacity, flow, residual):
    S_idx = [0]
    for u in S_idx:
        for v in range(len(names)):
            if v not in S_idx and residual[u, v] > 0:
                S_idx.append(v)

    node_colors = []
    for name in G.nodes():
        if names.index(name) in S_idx:
            node_colors.append("lightblue")
        else:
            node_colors.append("lightsalmon")

    edge_colors = []
    edge_widths = []
    edge_labels = {}
    cut_value = 0
    for (u, v) in G.edges():
        i = names.index(u)
        j = names.index(v)
        edge_labels[(u, v)] = str(flow[i, j]) + "/" + str(capacity[i, j])
        if i in S_idx and j not in S_idx:
            edge_colors.append("red")
            edge_widths.append(3.5)
            cut_value += capacity[i, j]
        else:
            edge_colors.append("gray")
            edge_widths.append(1.5)

    plt.figure(figsize=(8, 4.5))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           edgecolors="black", node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           arrowsize=20, node_size=900)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, label_pos=0.35, font_size=10)
    plt.title("Минимальный разрез = " + str(cut_value) +
              "\nсиние вершины - S, оранжевые - T, красные дуги - дуги разреза")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def draw_final(G, pos, names, capacity, flow, title="Максимальный поток"):
    edge_widths = []
    edge_labels = {}
    for (u, v) in G.edges():
        i = names.index(u)
        j = names.index(v)
        edge_labels[(u, v)] = str(flow[i, j]) + "/" + str(capacity[i, j])
        if flow[i, j] == capacity[i, j]:
            edge_widths.append(3.5)
        else:
            edge_widths.append(1.5)

    plt.figure(figsize=(8, 4.5))
    nx.draw_networkx_nodes(G, pos, node_color="white",
                           edgecolors="black", node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=edge_widths,
                           arrowsize=20, node_size=900)
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, label_pos=0.35, font_size=10)
    plt.title(f"{title} = " + str(flow[0, :].sum()))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_flow(names, capacity, log, flow, residual):
    G, pos = build_graph(names, capacity)

    draw_step(G, pos, names, capacity, np.zeros_like(capacity), None,
              "Начало: поток = 0")

    for k in range(len(log)):
        path, delta, value, flow_k = log[k]
        path_text = " → ".join(names[v] for v in path)
        title = ("Шаг " + str(k + 1) + ": " + path_text +
                 ", delta = " + str(delta) + ", поток = " + str(value))
        draw_step(G, pos, names, capacity, flow_k, path, title)

    draw_final(G, pos, names, capacity, flow)

    draw_cut(G, pos, names, capacity, flow, residual)


def flow_value(names, capacity):
    f, lg, res = max_flow(names, capacity)
    return f[0, :].sum() - f[:, 0].sum()


def plot_new_flow(names, new_data, title="Новый максимальный поток"):
    flow_new, log_new, residual_new = max_flow(names, new_data)
    G_new, pos_new = build_graph(names, new_data)
    draw_final(G_new, pos_new, names, new_data, flow_new, title=title)


def increase_flow(names, data_init):
    n = len(names)
    new_data = data_init.copy()
    while True:
        base = flow_value(names, new_data)
        found = False
        for i in range(n):
            for j in range(n):
                if new_data[i, j] > 0:
                    test = new_data.copy()
                    test[i, j] += 1
                    if flow_value(names, test) > base:
                        new_data = test
                        found = True
                        break
            if found:
                break
        if not found:
            break
    return new_data


# ================================================================================
# Часть 2. Числа Фибоначчи
# ================================================================================


def fibonacci(n):
    if n == 0 or n == 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_cache(n, fibonacci_dict):
    val = fibonacci_dict.get(n, -1)
    if val == -1:
        fibonacci_dict[n] = fibonacci_cache(
            n - 1, fibonacci_dict) + fibonacci_cache(n - 2, fibonacci_dict)
        val = fibonacci_dict[n]
    return val


def fibonacci_dynamic(n, is_print=False):
    if n < 2:
        return n
    a = 0
    b = 1
    if is_print:
        print(a)
        print(b)
    for i in range(n - 1):
        temp_a = a
        a = b
        b = b + temp_a
        if is_print:
            print(b)
    return b


def measure(func, repeats, series=1):
    best = None
    for s in range(series):
        start = time.perf_counter()
        for r in range(repeats):
            func()
        end = time.perf_counter()
        t = (end - start) / repeats
        if best is None or t < best:
            best = t
    return best


# ================================================================================
# Часть 3. Задача о рюкзаке
# ================================================================================


def backpack_gready(price, capacity, restriction):
    unit_cost = price / capacity
    indices = np.argsort(unit_cost)[::-1]
    current_capacity = 0
    current_price = 0
    taken = np.zeros_like(indices)
    for idx in indices:
        if current_capacity + capacity[idx] <= restriction:
            current_capacity += capacity[idx]
            current_price += price[idx]
            taken[idx] = 1
    return taken, current_capacity, current_price


def reconstruct_backpack(V, capacity, restriction):
    n = len(capacity)
    b = restriction
    taken = np.zeros(n)

    for k in range(n, 0, -1):
        if V[k, b] != V[k - 1, b]:
            taken[k - 1] = 1
            b -= capacity[k - 1]
    return taken


def backpack_dynamic(price, capacity, restriction):
    n = len(price)
    k_arr = np.arange(1, n + 1, 1)
    b_arr = np.arange(0, restriction + 1, 1)
    V = np.zeros((n + 1, restriction + 1), dtype=int)
    for b in b_arr:
        for k in k_arr:
            if capacity[k - 1] > b:
                V[k, b] = V[k - 1, b]
            else:
                V[k, b] = max(V[k - 1, b], price[k - 1] +
                              V[k - 1, b - capacity[k - 1]])
    taken = reconstruct_backpack(V, capacity, restriction)
    max_price = V[n, restriction]
    max_capacity = np.sum(capacity[taken.astype("bool")])
    return V, taken, max_price, max_capacity
