from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()

# Add nodes
stations = ["Лукянівська", "ЗолотіВорота", "Театральна", "Контрактова", "Поштова", "Майдан", "Хрещатик", "ПлощаУкрГероїв", "ПалацСпорту", "Кловська", "Печерська", "Олімпійська"]
G.add_nodes_from(stations)

# Add edges
connections = [("Лукянівська", "ЗолотіВорота", 5), ("ЗолотіВорота", "Театральна", 1), ("ПалацСпорту", "ЗолотіВорота", 3), ("Театральна", "Хрещатик", 3), ("ПлощаУкрГероїв", "ПалацСпорту", 1), 
                  ("ПалацСпорту",  "Кловська", 3), ( "Кловська", "Печерська", 3),  ("ПлощаУкрГероїв", "Олімпійська", 3), ("Майдан", "Поштова", 4),
                  ("ПлощаУкрГероїв", "Майдан", 3), ("Хрещатик", "Майдан", 1), ("Поштова", "Контрактова", 2)]
for connection in connections:
    G.add_edge(connection[0], connection[1], weight=connection[2])

# Visualize the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=800, node_color='skyblue', edge_color='gray', linewidths=0.5)
plt.title('Kyiv Metro Stations Network', fontsize=16)
plt.show()

print(f"кількість вершин {len(G.nodes())}")
print(f"кількість ребер {len(G.edges())}")

# Degree of each node
degrees = dict(G.degree())
print("Degree of each station:")
for station in stations:
    print(f"{station}: {degrees[station]}")

def dfs(graph, start, goal):
    visited = set()
    stack = [(start, [start])]

    while stack:
        node, path = stack.pop()
        if node not in visited:
            visited.add(node)

            if node == goal:
                return path

            for neighbor in graph.neighbors(node):
                stack.append((neighbor, path + [neighbor]))

    return None

# Function for BFS
def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        if node not in visited:
            visited.add(node)

            if node == goal:
                return path

            for neighbor in graph.neighbors(node):
                queue.append((neighbor, path + [neighbor]))

    return None

# Find paths using DFS and BFS
start_station = "Лукянівська"
end_station = "Олімпійська"

dfs_path = dfs(G, start_station, end_station)
bfs_path = bfs(G, start_station, end_station)

# Print the results
print(f"DFS Path from {start_station} to {end_station}: {dfs_path}")
print(f"BFS Path from {start_station} to {end_station}: {bfs_path}")

def dijkstra(graph, start, end):
    distances = {node: float('infinity') for node in graph.nodes}
    distances[start] = 0
    visited = set()

    while visited != set(graph.nodes):
        current_node = None
        min_distance = float('infinity')
        for node in set(graph.nodes) - visited:
            if distances[node] < min_distance:
                current_node = node
                min_distance = distances[node]

        visited.add(current_node)
        neighbors = set(graph.neighbors(current_node)) - visited
        for neighbor in neighbors:
            potential_distance = distances[current_node] + graph.get_edge_data(current_node, neighbor)['weight']
            if potential_distance < distances[neighbor]:
                distances[neighbor] = potential_distance

    path = [end]
    while end != start:
        for neighbor in graph.neighbors(end):
            if graph.get_edge_data(end, neighbor) != None and (distances[end] == distances[neighbor] + graph.get_edge_data(end, neighbor)['weight']):
                path.append(neighbor)
                end = neighbor

    return path[::-1], distances

# Find the shortest path using Dijkstra's algorithm
start_station = "Лукянівська"
end_station = "Олімпійська"

shortest_path, shortest_distance = dijkstra(G, start_station, end_station)
print(f"Коротший шлях з {start_station} до {end_station}: {shortest_path}")
print(f"Корортша дистанція: {shortest_distance[end_station]}")
